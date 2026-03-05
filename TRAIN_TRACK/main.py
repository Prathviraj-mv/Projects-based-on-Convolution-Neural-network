import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from datetime import datetime
from collections import Counter

train_loc = "combined_train2"
test_loc = "combined_test2"

device = torch.device("cuda")

train_transform = transforms.Compose([
    transforms.CenterCrop((640,220)),
    transforms.Resize((256,256)),
    transforms.GaussianBlur(3),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.CenterCrop((640, 220)),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]
                         )
])

train_data = datasets.ImageFolder(root=train_loc,
                                  transform=train_transform
                                  )
test_data = datasets.ImageFolder(root=test_loc,
                                 transform=val_transform
                                 )

train = DataLoader(train_data,
                   batch_size=32,
                   shuffle=True,
                   num_workers=0,
                   pin_memory=True
                   )
test = DataLoader(test_data,
                  batch_size=32,
                  shuffle=False,
                  num_workers=0,
                  pin_memory=True
                  )

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

length = len(train_data.classes)
model.classifier[1] = nn.Linear(model.last_channel, length)

for param in model.features.parameters():
    param.requires_grad = False

model = model.to(device)

class_counts = Counter(train_data.targets)
weights = torch.tensor([1.0/class_counts[i] for i in range(length)], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0003)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.3)

epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(datetime.now())

    for image, label in train:
        image = image.to(device)
        label = label.to(device)

        output = model(image)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")


torch.save({
    "model_state": model.state_dict(),
    "classes": train_data.classes
}, "rail_defect_model.pth")
