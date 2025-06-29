import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

transform = weights.transforms()
labels = weights.meta["categories"]

def classify_image(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        _, predicted = output.max(1)
        return labels[predicted.item()]
