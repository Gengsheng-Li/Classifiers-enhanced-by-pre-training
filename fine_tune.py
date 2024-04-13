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

def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier on top of CLIP embeddings.')
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
                        help='Fine-tuning mode')
    
    
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

def training_validating_testing(classifier, criterion, optimizer, trainloader, valloader, testloader):
    # 开始训练
    for epoch in tqdm(range(args.epochs)):
        total_loss, total_val_loss = 0, 0
        
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
        
        # 验证阶段
        num_val, correct_val = 0, 0
        
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
        val_accuracy = 100 * correct_val / num_val
        wandb.log({"epoch": epoch + 1, "average_loss": average_loss, "average_val_loss": average_val_loss, "val_accuracy": val_accuracy})
        print(f"Epoch: {epoch + 1}, Average_loss: {average_loss}, Average_val_loss: {average_val_loss}, Val_accuracy: {val_accuracy}")

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
            },
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    trainloader, valloader, testloader = dataloader(args)

    training_validating_testing(classifier, criterion, optimizer, trainloader, valloader, testloader)

if __name__ == '__main__':
    args = parse_args()
    main(args)