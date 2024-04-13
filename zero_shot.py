import torch
import clip
import wandb
from tqdm import tqdm
from torchvision import datasets, transforms

wandb.login()
wandb.init(
    project='clip-linear-probe-classification', 
    )

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 准备数据
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=preprocess)
testloader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=False)

# CIFAR-100类别描述
class_descriptions = [f"This is a photo of a {label}." for label in testset.classes]
text_tokens = clip.tokenize(class_descriptions).to(device)

# Zero-shot分类
total_correct = 0
with torch.no_grad():
    for images, labels in tqdm(testloader):
        images, labels = images.to(device), labels.to(device)
        # 编码并计算相似度
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)

        # 使用余弦相似度计算匹配度
        similarities = (image_features @ text_features.T).softmax(dim=-1)
        predictions = similarities.argmax(dim=-1)

        total_correct += (predictions == labels).sum().item()

# 计算准确率
zero_shot_accuracy = total_correct / len(testset)
wandb.log({"test_accuracy": zero_shot_accuracy * 100})
print(f"CLIP Zero-Shot Accuracy on CIFAR-100: {zero_shot_accuracy * 100:.2f}%")