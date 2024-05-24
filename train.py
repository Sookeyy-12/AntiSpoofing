from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="Datasets/SplitData/data.yaml",
            epochs = 3,
)