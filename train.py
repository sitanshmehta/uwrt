from ultralytics import YOLO

path = "/Users/sitanshmehta/Desktop/Keyboard Detection v2.v1i.yolov11/data.yaml"

model = YOLO("yolo11n.pt") 

results = model.train(data=path, epochs=20, imgsz=640, device='cpu')
