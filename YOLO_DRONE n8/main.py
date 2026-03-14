from ultralytics import YOLO
import torch

def main():

    print("GPU:", torch.cuda.is_available())

    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        device=0,
        verbose=True
    )

if __name__ == "__main__":
    main()
