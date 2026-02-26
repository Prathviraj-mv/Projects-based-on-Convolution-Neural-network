import torch
import datetime
from torchvision.models import mobilenet_v2
from torchvision import transforms
import torch.nn as nn

print(datetime.datetime.now())
model =mobilenet_v2(weights =None)
model.classifier[1] = nn.Linear(model.last_channel, 10)
model.load_state_dict(torch.load("hand_model.pth", map_location="cpu"))
model.eval()

classes = ["0","1","2","3","4","5","6","7","8","9"]
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
print(next(model.parameters()).sum())
import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    img = transform(frame).unsqueeze(0)

    with torch.no_grad():
        pred = torch.argmax(model(img),1).item()

    cv2.putText(frame, classes[pred], (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1)==27:
        break
