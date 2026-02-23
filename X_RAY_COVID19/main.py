import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import datetime

device =torch.device("cuda")
train_transformer = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

test_transformer = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)

training_data =datasets.ImageFolder(root="train",transform=train_transformer)
testing_data =datasets.ImageFolder(root="test",transform=test_transformer)

train_loader =DataLoader(training_data,batch_size=64,shuffle=True)
test_loader =DataLoader(testing_data,batch_size=64,shuffle=False)

model = nn.Sequential(

    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),

    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(128, 2)
)


model =model.to(device = device)
optimiser = torch.optim.Adam(model.parameters(),lr =0.001)
criterion = nn.CrossEntropyLoss()

epochs = 10


for epoch in range(epochs):
    print(datetime.datetime.now())
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print("Test Accuracy:", accuracy)
