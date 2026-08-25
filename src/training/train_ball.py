import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

from ultralytics import YOLO
import torch

if __name__ == "__main__":
    torch.cuda.empty_cache()  # clear before starting
    
    model = YOLO("yolo26x.pt")

    results = model.train(
        data="data/ball/data.yaml",
        epochs=150,
        imgsz=1280,
        batch=2,
        workers=0,
        patience=40,
        save=True,
        save_period=10,
        project="runs/ball",
        name="yolo26x_v3",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.001,
        weight_decay=0.001,
        warmup_epochs=8,
        mosaic=0.0,          # OFF entirely — this was destroying small ball features
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=0,
        amp=True,
        cache=False,
        box=12.0,
        cls=0.2,
        dropout=0.1,
        hsv_h=0.02,          # use colour jitter instead of mosaic for augmentation
        hsv_s=0.5,
        hsv_v=0.3,
        fliplr=0.5,
        translate=0.1,
        scale=0.3,           # mild scale augmentation — simulates different ball distances
    )

    print(f"Training complete. Best weights: {results.save_dir}/weights/best.pt")