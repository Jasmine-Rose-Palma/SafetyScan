import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import PPE_CNN

# CONFIG
DATA_DIR = "data/cnn_dataset"   # adjust if needed
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# DATASET
train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=transform)
val_dataset = datasets.ImageFolder(f"{DATA_DIR}/valid", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# MODEL
model = PPE_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# TRAIN LOOP
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}")

# SAVE MODEL
torch.save(model.state_dict(), "models/cnn_model.pth")
print("CNN model saved!")