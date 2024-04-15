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
from model import CLIPClassifier_for_tuning

def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier on top of CLIP embeddings.')
    # Use or forbid Weights & Biases (recommand to use it)
    parser.add_argument('--forbid_wandb', action='store_true', help='Set this flag to forbid Weights & Biases')

    # Arguments for fine-tuning 
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of the original training data to use for validation')
    parser.add_argument('--train_mode', type=str, default='linear-probe', choices=['linear-probe', 'fine-tune'], 
                        help='Training mode')
    parser.add_argument('--save_path', type=str, default="results",
                        help='Path for saving model weights.')
    parser.add_argument('--exp_name', type=str, default="train_from_pretrain",
                        help='Experiment name.')
    
    # Arguments for initializing CLIPClassification
    parser.add_argument('--num_classes', type=int, default=100,
                        help='The number of classes to be classified. As the dataset used here is CIFAR-100, so the default value is 100. \
                            However, you can choose any other value as long as it fits your dataset.')
    
    return parser.parse_args()

def dataloader(args, preprocess):
    # 加载CIFAR-100训练和测试数据集
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


def training_validating_testing(args, classifier, criterion, optimizer, trainloader, valloader, testloader):
    # 开始训练
    for epoch in tqdm(range(args.epochs)):
        num_trn, correct_trn, total_loss = 0, 0, 0
        
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
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct_trn += (predicted == labels).sum().item()
            num_trn += labels.size(0)
        
        # 记录、更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        # scheduler.step()
        
        # 验证阶段
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
        average_loss = total_loss / len(trainloader)
        average_val_loss = total_val_loss / len(valloader)
        trn_accuracy = 100 * correct_trn / num_trn
        val_accuracy = 100 * correct_val / num_val
        wandb.log({"epoch": epoch, "average_loss": average_loss, "trn_accuracy": trn_accuracy, "average_val_loss": average_val_loss, "val_accuracy": val_accuracy, "lr": current_lr})
        print(f"Epoch: {epoch}, Average Loss: {average_loss}, Trn Accuracy: {trn_accuracy:.2f}%, lr: {current_lr}")
        print(f"Average Val Loss: {average_val_loss}, Val Accuracy: {val_accuracy}")

        # 保存模型
        if (epoch) % 10 == 0:
            # save_dir = os.path.join(args.save_path, args.exp_name, f"lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}_ep{args.epochs}_T{args.T_max}")
            save_dir = os.path.join(args.save_path, args.exp_name, f"{args.train_mode}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}")
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
            "train_mode": args.train_mode,
            },
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载预训练的CLIP模型
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # 获取CLIP模型中的视觉编码器
    vis_encoder = clip_model.visual

    # 初始化用于微调的分类器，使用CLIP的视觉编码器作为预训练编码器
    classifier = CLIPClassifier_for_tuning(args, pretrained_encoder=vis_encoder).to(device)

    # 根据训练模式设置模型参数和优化器
    if args.train_mode == "linear-probe":
        # 冻住CLIP的视觉编码器，这意味着在训练过程中不更新编码器的权重
        for param in classifier.encoder.parameters():
            param.requires_grad = False
        
        # 仅设置分类器最后的全连接层参数进行训练
        optimizer = torch.optim.Adam(classifier.fc.parameters(), lr=args.lr)

    elif args.train_mode == "fine-tune":
        # 对整个模型的所有参数进行更新，即视觉编码器和分类层都将更新
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    else:
        # 如果train_mode设置不正确，则抛出异常
        raise ValueError("Invalid training mode specified. Choose either 'linear-probe' or 'fine-tune'.")

    
    criterion = nn.CrossEntropyLoss()

    trainloader, valloader, testloader = dataloader(args, preprocess)

    training_validating_testing(args, classifier, criterion, optimizer, trainloader, valloader, testloader)

if __name__ == '__main__':
    args = parse_args()
    if args.forbid_wandb:
        os.environ['WANDB_DISABLED'] = 'true'
        
    main(args)