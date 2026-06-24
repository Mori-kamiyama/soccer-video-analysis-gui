from ultralytics import YOLO
model = YOLO("models/yolo11m.pt")
model.train(
    data="dataset/data.yaml",
    epochs=80,
    imgsz=1280,
    batch=6,
    device="mps",
    patience=20,
    project="runs", name="player_ft",
    exist_ok=True,
    # 単一試合ドメイン: 色変化は控えめ、幾何は軽く
    hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,
    degrees=0.0, translate=0.05, scale=0.2, fliplr=0.5,
    mosaic=1.0, close_mosaic=10,
)
