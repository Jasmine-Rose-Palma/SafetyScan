import torch
import torchvision.transforms as transforms
import cv2
from cnn.model import PPE_CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PPE_CNN().to(device)
model.load_state_dict(torch.load("models/cnn_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

classes = ["head", "helmet"]

def classify_crop(crop):
    image = transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]