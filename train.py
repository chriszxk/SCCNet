from ultralytics import YOLO
import torch

model = YOLO("/mnt/DATA/quabot/work/github/HSCD/ultralytics/cfg/models/SCCNet_files/SCCNet.yaml")
train_results = model.train(
    data="yourdata.yaml",
    epochs=500,
    imgsz=640,
    device=1,
    project="your save path",
    name="train",
    resume=True,
    workers=5,
)