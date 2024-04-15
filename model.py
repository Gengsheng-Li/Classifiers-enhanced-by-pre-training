
from torch import nn
from clip.model import VisionTransformer

class CLIPClassifier(nn.Module):
    def __init__(self, args):
        super(CLIPClassifier, self).__init__()
        self.encoder = VisionTransformer(input_resolution=args.input_resolution, patch_size=args.patch_size, width=args.width, layers=args.layers, heads=args.heads, output_dim=args.output_dim)
        self.fc = nn.Linear(self.encoder.output_dim, args.num_classes)

    def forward(self, x):
        embedding = self.encoder(x)
        return self.fc(embedding)
    
class CLIPClassifier_for_tuning(nn.Module):
    def __init__(self, args, pretrained_encoder):
        super(CLIPClassifier_for_tuning, self).__init__()
        self.encoder = pretrained_encoder.float()
        self.fc = nn.Linear(self.encoder.output_dim, args.num_classes)

    def forward(self, x):
        embedding = self.encoder(x)
        return self.fc(embedding)