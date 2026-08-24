import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

from ultralytics import YOLO
import torch

if __name__ == "__main__":
    torch.cuda.empty_cache()  # clear before starting
    
    model = YOLO("yolo26x.pt")

    results = model.train(
        data="data/ball/data.yaml",
        epochs=100,
        imgsz=1280,
        batch=2,
        workers=0,          # back to 0 — 8 workers at 1280 is eating VRAM
        patience=20,
        save=True,
        save_period=5,      # save checkpoint every 5 epochs so you don't lose progress
        project="runs/ball",
        name="yolo26x",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0005,
        mosaic=1.0,
        mixup=0.05,
        conf=0.25,
        amp=True,           # Automatic Mixed Precision — cuts VRAM usage significantly
        cache=False,        # turn cache OFF — at 1280 it's consuming too much RAM
    )

    print(f"Training complete. Best weights: {results.save_dir}/weights/best.pt")