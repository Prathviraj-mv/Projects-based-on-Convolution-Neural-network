import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import datetime

gpu = "cuda"
device =torch.device(gpu)

train_transforms =transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
]

)

test_transformer  =transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),

])

train_folder = datasets.ImageFolder(root ="Training",transform=train_transforms)
test_folder = datasets.ImageFolder(root ="Testing",transform=test_transformer)

data_train = DataLoader(train_folder,shuffle=True,batch_size=32)
data_test = DataLoader(test_folder,shuffle=False,batch_size=32)

# sanity test
image,label =train_folder[0]
print(image.shape)
print(label)

model =nn.Sequential(
    nn.Conv2d(in_channels=3,kernel_size=5,out_channels=64),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(in_channels=64, kernel_size=5, out_channels=128),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(in_channels=128, kernel_size=5, out_channels=256),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(in_channels=256, kernel_size=5, out_channels=512),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(512,len(train_folder.classes))


)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr =0.001)
criterion = nn.CrossEntropyLoss()

epoch =5
for epoch in range(epoch):
    model.train()
    total_loss = 0
    print(datetime.datetime.now())

    for image, label in data_train:
        image = image.to(device)
        label = label.to(device)

        output = model(image)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for image,label in data_test:

        image = image.to(device)
        label = label.to(device)

        output = model(image)
        pred = torch.argmax(output,dim=1)

        correct += (pred == label).sum().item()
        total += label.size(0)

accuracy = 100*correct/total
print(f"Test Accuracy: {accuracy:.2f}%")



# torch.Size([3, 224, 224])
# 0
# 2026-03-15 11:07:22.026995
# Epoch 1, Loss: 123.5736
# 2026-03-15 11:08:31.260654
# Epoch 2, Loss: 107.7128
# 2026-03-15 11:09:12.292493
# Epoch 3, Loss: 99.0296
# 2026-03-15 11:09:48.966369
# Epoch 4, Loss: 92.4446
# 2026-03-15 11:10:34.885587
# Epoch 5, Loss: 87.0486
# Test Accuracy: 68.93%

# Process finished with exit code 0
