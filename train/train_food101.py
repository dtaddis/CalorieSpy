# train/train_food101.py

import torch
from torch import nn, optim
from torchvision import transforms, models
from torchvision.datasets import Food101
from torch.utils.data import DataLoader
import os
import time
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📦 Training on {device}")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load datasets
train_set = Food101(root='data', split='train', transform=transform, download=True)
test_set = Food101(root='data', split='test', transform=transform, download=True)

# Save label list for use in classifier.py
label_path = os.path.join("calorie_data", "food101_labels.txt")
os.makedirs("calorie_data", exist_ok=True)
with open(label_path, "w") as f:
    for label in sorted(train_set.classes):
        f.write(label + "\n")

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32)

# Load pretrained model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 101)  # Food-101 has 101 classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# Training loop
epochs = 15
for epoch in range(epochs):
    start_time = time.time()
    print(f"🕒 Epoch {epoch+1}/{epochs} started at {datetime.now().strftime('%H:%M:%S')}")
    
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    end_time = time.time()
    duration = end_time - start_time
    print(f"✅ Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Completed at {datetime.now().strftime('%H:%M:%S')} (⏱️ {duration:.2f}s)")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"📊 Test Accuracy: {100 * correct / total:.2f}%")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/food101_resnet50.pth")
print("🎉 Model saved to models/food101_resnet50.pth")
