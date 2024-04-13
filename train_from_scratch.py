import os
import torch
import wandb
import clip
import argparse
from torch import nn
from tqdm import tqdm
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from CLIPClassifier_model import CLIPClassifier
from torch.optim.lr_scheduler import CosineAnnealingLR

def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier on top of CLIP embeddings.')
    # Arguments for training 
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--T_max', type=int, default=100, 
                        help='Maximum number of iterations for cosine annealing')
    parser.add_argument('--weight_decay', type=float, default=0.2,
                        help='Weight decay for non-bias and non-gain parameters')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of the original training data to use for validation')
    parser.add_argument('--save_path', type=str, default="results",
                        help='Path for saving model weights.')
    parser.add_argument('--exp_name', type=str, default="train_from_scratch",
                        help='Experiment name.')
    
    # Arguments for initializing CLIPClassification
    parser.add_argument('--num_classes', type=int, default=100,
                        help='The number of classes to be classified. As the dataset used here is CIFAR-100, so the default value is 100. \
                            However, you can choose any other value as long as it fits your dataset.')
    parser.add_argument('--input_resolution', type=int, default=224,
                        help='Input resolution of image encoder')
    parser.add_argument('--output_dim', type=int, default=512,
                        help='Dimension of the embedding output by the encoder')
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    
    return parser.parse_args()

def dataloader(args):
    # 加载CIFAR-100训练和测试数据集
    _, preprocess = clip.load("ViT-B/32", device='cuda')  # Load the official image preprocessing pipeline from OpenAI's CLIP, while the model is ignored.
    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=preprocess)
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=preprocess)

    # 划分训练集和验证集
    train_indices, val_indices = train_test_split(list(range(len(trainset))), test_size=args.validation_split, stratify=trainset.targets)
    train_subset = Subset(trainset, train_indices)
    val_subset = Subset(trainset, val_indices)

    # 创建数据加载器
    trainloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    return trainloader, valloader, testloader

def training_validating_testing(classifier, criterion, optimizer, scheduler, trainloader, valloader, testloader):
    # 开始训练
    for epoch in tqdm(range(args.epochs)):
        total_loss = 0
        
        # 训练阶段
        classifier.train()
        for images, labels in trainloader:
            images, labels = images.cuda(), labels.cuda()
            
            # 通过CLIP模型获取图像特征
            outputs = classifier(images)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()  # 累积每个batch的loss
        
        # 记录、更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # 记录指标
        average_loss = total_loss / len(trainloader)
        wandb.log({"epoch": epoch, "average_loss": average_loss, "lr": current_lr})
        print(f"Epoch: {epoch}, Average_loss: {average_loss}, lr: {current_lr}")
        
        # 验证阶段
        if (epoch) % 2 == 0:
            print("### Validation ###")            
            num_val, correct_val, total_val_loss = 0, 0, 0
            
            classifier.eval()
            with torch.no_grad():
                for images, labels in valloader:
                    images, labels = images.cuda(), labels.cuda()
                    outputs = classifier(images)
                    val_loss = criterion(outputs, labels)
                    total_val_loss += val_loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct_val += (predicted == labels).sum().item()
                    num_val += labels.size(0)

            # 记录指标
            average_val_loss = total_val_loss / len(valloader)
            val_accuracy = 100 * correct_val / num_val
            wandb.log({"average_val_loss": average_val_loss, "val_accuracy": val_accuracy})
            print(f"Average_val_loss: {average_val_loss}, Val_accuracy: {val_accuracy}")

        # 保存模型
        if (epoch) % 10 == 0:
            save_dir = os.path.join(args.save_path, args.exp_name, f"lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}_ep{args.epochs}_T{args.T_max}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(classifier.state_dict(), f"{save_dir}/classifier_epoch_{epoch}.pth")

    # 测试阶段
    num_test, correct_test = 0, 0

    classifier.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            num_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    # 记录指标
    wandb.log({"test_accuracy": 100 * correct_test / num_test})
    print(f'Accuracy on the 10000 test images: {100 * correct_test / num_test}%')

def main(args):
    wandb.login()
    wandb.init(
        project='clip-linear-probe-classification', 
        config = {
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "validation_split": args.validation_split,
            "T_max": args.T_max,
            "weight_decay": args.weight_decay},
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = CLIPClassifier(args).to(device)
    
    # Separate parameters for weight decay
    decay_params = []  # Parameters to apply weight decay
    no_decay_params = []  # Parameters that should not have weight decay
    
    for name, param in classifier.named_parameters():
        if not name.endswith(".bias") and not 'gain' in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    # Optimizer with different weight decay settings for different parameter groups
    optimizer = torch.optim.Adam([
        {'params': no_decay_params, 'weight_decay': 0.0},  # No weight decay for biases and gains
        {'params': decay_params, 'weight_decay': args.weight_decay}  # Weight decay for other parameters
        ], lr=args.lr)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max)

    trainloader, valloader, testloader = dataloader(args)

    training_validating_testing(classifier, criterion, optimizer, scheduler, trainloader, valloader, testloader)

if __name__ == '__main__':
    args = parse_args()
    main(args)