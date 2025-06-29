import torch
from torchvision import models, transforms
from PIL import Image

# Load ImageNet-pretrained ResNet18
model = models.resnet18(pretrained=True)
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ImageNet class index to human-readable labels
with open("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

def classify_image(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(tensor)
        _, predicted = output.max(1)
        return labels[predicted.item()]
