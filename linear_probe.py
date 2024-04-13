import torch
import wandb
import clip
import argparse
from torch import nn
from tqdm import tqdm
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader



def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier on top of CLIP embeddings.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of the original training data to use for validation')
    
    return parser.parse_args()

args = parse_args()
lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
validation_split = args.validation_split

wandb.login()
wandb.init(
    project='clip-linear-probe-classification', 
    config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "validation_split": validation_split,
        },
    )



# 加载预训练的CLIP模型
model, preprocess = clip.load("ViT-B/32", device='cuda')

# 添加一个线性分类头
num_classes = 100  # CIFAR-100类别数
classifier = nn.Linear(model.visual.output_dim, num_classes).to('cuda')

# 固定CLIP模型的参数
for param in model.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()



# 加载CIFAR-100训练和测试数据集
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=preprocess)
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=preprocess)

# 划分训练集和验证集
train_indices, val_indices = train_test_split(list(range(len(trainset))), test_size=validation_split, stratify=trainset.targets)
train_subset = Subset(trainset, train_indices)
val_subset = Subset(trainset, val_indices)

# 创建数据加载器
trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)



# 开始训练
for epoch in tqdm(range(epochs)):
    total_loss, total_val_loss, correct_val = 0, 0, 0
    
    # 训练阶段
    classifier.train()
    for images, labels in trainloader:
        images, labels = images.cuda(), labels.cuda()
        
        # 通过CLIP模型获取图像特征
        with torch.no_grad():
            image_features = model.encode_image(images)
        
        # 通过分类头进行预测
        outputs = classifier(image_features.float())
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()  # 累积每个batch的loss
    
    # 验证阶段
    classifier.eval()
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.cuda(), labels.cuda()
            image_features = model.encode_image(images)
            outputs = classifier(image_features.float())
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
    
    average_loss = total_loss / len(trainloader)
    average_val_loss = total_val_loss / len(valloader)
    val_accuracy = 100 * correct_val / len(val_subset)
    wandb.log({"epoch": epoch + 1, "average_loss": average_loss, "average_val_loss": average_val_loss, "val_accuracy": val_accuracy})
    print(f"Epoch: {epoch + 1}, Average_loss: {average_loss}, Average_val_loss: {average_val_loss}, Val_accuracy: {val_accuracy}")


# 测试阶段
total = 0
correct = 0

classifier.eval()
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.cuda(), labels.cuda()
        image_features = model.encode_image(images)
        outputs = classifier(image_features.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

wandb.log({"test_accuracy": 100 * correct / total})
print(f'Accuracy on the 10000 test images: {100 * correct / total}%')