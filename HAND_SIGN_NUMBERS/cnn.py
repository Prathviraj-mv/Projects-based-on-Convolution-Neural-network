import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import datetime

device = torch.device("cuda")
print(torch.cuda.get_device_name(0))


train_transformer =transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(15),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train = datasets.ImageFolder(root="IMAGES",transform=train_transformer)
train_data =DataLoader(train,batch_size=32,shuffle=True)


from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, len(train.classes))
model = model.to(device)
for param in model.features.parameters():
    param.requires_grad = False

criterion =nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(),lr=0.00005)




epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(datetime.datetime.now())
    for images, labels in train_data:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


torch.save(model.state_dict(), "hand_model.pth")
