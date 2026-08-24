from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26x-pose.pt") 

    results = model.train(
        data="data/field/data.yaml",
        epochs=50,
        imgsz=1280,
        batch=3,
        patience=20,
        save=True,
        project="runs/field",
        name="yolo26x_pose",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        mosaic=0.5,
    )

    print(f"Training complete. Best weights: {results.save_dir}/weights/best.pt")