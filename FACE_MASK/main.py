import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18,ResNet18_Weights
from datetime import datetime
import torch.nn as nn
transforms =transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train = datasets.ImageFolder(
    root="faceMASK/data",
    transform = transforms,
)
test = datasets.ImageFolder(
    root="faceMASK/eval_",
    transform = transforms,
)

train_l = DataLoader(train,batch_size=64,shuffle=False)
test_l = DataLoader(test,batch_size=64,shuffle=True)

image,label =train[0]
device =torch.device("cuda")

print(image.shape)
print(label)
print(train.classes)
print(device)
Weights =ResNet18_Weights.DEFAULT
model =resnet18(weights =Weights)


length = len(train.classes)
model.fc = nn.Linear(model.fc.in_features, length)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 5

model.train()

for epoch in range(epochs):
    time = datetime.now().strftime("%M:%S")
    for images, labels in train_l:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)    # forward pass
        loss = criterion(outputs, labels)  # compute error
        loss.backward()            # backprop
        optimizer.step()           # update weights

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


image, label = train[0]
image = image.unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    output = model(image)
    prediction = torch.argmax(output, dim=1).item()


print("True label:", label)
print("Predicted:", prediction)
