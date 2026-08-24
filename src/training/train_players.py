from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26x.pt")

    results = model.train(
        data="data/players/data.yaml",
        epochs=50,
        imgsz=1280,
        batch=3,
        patience=20,
        save=True,
        project="runs/players",
        name="yolo26x",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    print(f"Training complete. Best weights: {results.save_dir}/weights/best.pt")