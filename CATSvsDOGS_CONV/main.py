import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
device = torch.device("cuda")

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

train =datasets.ImageFolder(root="pets/train",transform =transform,)
test =datasets.ImageFolder(root="pets/test",transform =transform,)
train_l =DataLoader(train,64,True)
test_l =DataLoader(test,64,False)
print(train.classes)

import torch.nn as nn
alexnet = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
).to(device)


criterion  = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(alexnet.parameters(),lr=3e-4)

epochs = 10


for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    alexnet.train()

    for images, labels in train_l:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = alexnet(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {total_loss/len(train_l):.4f} | "
        f"Acc: {100*correct/total:.2f}%"
    )

alexnet.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_l:
        images = images.to(device)
        labels = labels.to(device)

        outputs = alexnet(images)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")







