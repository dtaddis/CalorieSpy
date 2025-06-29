import torch
from torchvision import models, transforms
from PIL import Image
import os

# --- Load fine-tuned Food101 model ---
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 101)
model.load_state_dict(torch.load("models/food101_resnet50.pth", map_location="cpu"))
model.eval()

# Load class labels
with open("calorie_data/food101_labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Image transforms (simple resize + tensor for now)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def classify_image(image_path, topk=3):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top_probs, top_idxs = torch.topk(probs, topk)
        return [(labels[i], top_probs[j].item()) for j, i in enumerate(top_idxs)]
