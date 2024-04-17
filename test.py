import os
import torch
import argparse
import clip
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from model import CLIPClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier on top of CLIP embeddings.')
    # Use or forbid Weights & Biases (recommand to use it)
    parser.add_argument('--forbid_wandb', action='store_true', help='Set this flag to forbid Weights & Biases')

    # Arguments for chossing model
    parser.add_argument('--model_pth', type=str, default='results/fine-tune-best.pth',
                        help='The path of the model used for inference')
    parser.add_argument('--check_model', action='store_true',
                        help='Add this argument to check the model you choosed and print some related info layer by layer')
    parser.add_argument('--zero_shot', action='store_true',
                        help='Add this argument to use the pre-trained CLIP for classification without any further traning')
    
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

def count_files(directory, prefix):
    """Count files in the directory with a specific prefix."""
    return sum(1 for f in os.listdir(directory) if f.startswith(prefix) and os.path.isfile(os.path.join(directory, f)))

def main(args):
    wandb.login()
    wandb.init(
        project='clip-linear-probe-classification', 
        config = {
            "zero_shot": args.zero_shot,
            "model_pth": args.model_pth,
            },
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.zero_shot:
        print("NOTE: You are using zero-shot mode of pre-trained CLIP ViT-B/32 for classification!")
        
        model, preprocess = clip.load("ViT-B/32", device=device)

        # 加载CIFAR-100 test dataset
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=preprocess)
        testloader = DataLoader(testset, batch_size=64, shuffle=False)
        
        # 利用text template构建CIFAR-100类别描述
        class_descriptions = [f"This is a photo of a {label}." for label in testset.classes]
        text_tokens = clip.tokenize(class_descriptions).to(device)
        
        # Zero-shot分类
        all_labels, all_preds = [], []
        num_test, num_correct = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(testloader):
                images, labels = images.to(device), labels.to(device)
                
                # 编码
                image_features = model.encode_image(images)
                text_features = model.encode_text(text_tokens)

                # 使用余弦相似度计算匹配度
                similarities = (image_features @ text_features.T).softmax(dim=-1)
                predictions = similarities.argmax(dim=-1)

                num_correct += (predictions == labels).sum().item()
                num_test += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())
                
        # 计算并记录Accuracy、Precision、Recall、F1 Score
        print(f"CLIP Zero-Shot Accuracy on CIFAR-100 Test Dadaset: {100 * num_correct / num_test}%")
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")   
        
        wandb.log({
            "Accuracy": 100 * num_correct / num_test,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
        })  

        # 计算并保存混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(20, 20))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')

        output_dir = 'results/test/'
        os.makedirs(output_dir, exist_ok=True)
        file_count = count_files(output_dir, 'confusion_matrix')
        output_path = os.path.join(output_dir, f'confusion_matrix_{file_count}.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Confusion matrix saved to {output_path}")
    
    else:
        model_pth = args.model_pth
        state_dict = torch.load(model_pth, map_location=torch.device(device))
        print(f"The model you choosed is: {model_pth}")

        # 创建模型实例并加载状态字典
        model = CLIPClassifier(args).to(device)
        model.load_state_dict(state_dict)

        # 计算参数总数
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of parameters in model: {total_params}")
        
        # 检查模型
        if args.check_model:
            print("Checking...")
            for name, param in model.named_parameters():
                print(f"Layer: {name} | Size: {param.size()} | Type: {param.dtype}")

            all_float32 = all(param.dtype == torch.float32 for _, param in model.named_parameters())
            print("All parameters are float32:", all_float32)

        # 加载CIFAR-100 test dataset
        _, preprocess = clip.load("ViT-B/32", device='cuda')  # Load the official image preprocessing pipeline from OpenAI's CLIP, while the model is ignored.
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=preprocess)
        testloader = DataLoader(testset, batch_size=64, shuffle=False)
        
        # 在CIFAR-100 test dataset上进行分类测试
        all_labels, all_preds = [], []
        num_test, num_correct = 0, 0

        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(testloader):
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                num_test += labels.size(0)
                num_correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        # 计算并记录Accuracy、Precision、Recall、F1 Score
        print(f'CLIP-based Classifier Accuracy on CIFAR-100 Test Dadaset: {100 * num_correct / num_test}%')
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")   
        
        wandb.log({
            "Accuracy": 100 * num_correct / num_test,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
        })  

        # 计算并保存混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(20, 20))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')

        output_dir = 'results/test/'
        os.makedirs(output_dir, exist_ok=True)
        file_count = count_files(output_dir, 'confusion_matrix')
        output_path = os.path.join(output_dir, f'confusion_matrix_{file_count}.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Confusion matrix is saved to {output_path}")


if __name__ == '__main__':
    args = parse_args()
    if args.forbid_wandb:
        os.environ['WANDB_DISABLED'] = 'true'
    
    main(args)    