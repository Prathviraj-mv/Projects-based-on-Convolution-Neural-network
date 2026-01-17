import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


transform =transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

train =datasets.ImageFolder(
    root="MNIST Dataset JPG format/MNIST - JPG - training",
    transform=transform
)

test =datasets.ImageFolder(
    root="MNIST Dataset JPG format/MNIST - JPG - testing",
    transform=transform
)

train_l =DataLoader(train,batch_size=64,shuffle=True)
test_l =DataLoader(test,batch_size=64,shuffle=True)

image,label =train[0]
print(label)
print(image.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

import torch.nn as nn


model =nn.Sequential(
    nn.Conv2d(1,32,3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)


criterion =nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5


for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    model.train()

    for images, labels in train_l:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
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

image, label = test[0]

with torch.no_grad():
    output = model(image.unsqueeze(0))
    prediction = torch.argmax(output, dim=1).item()

print(prediction)
print(label)

plt.figure(figsize=(10,15))
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Predicted: {prediction}")
plt.show()
