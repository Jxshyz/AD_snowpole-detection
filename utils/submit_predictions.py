from ultralytics import YOLO
from pathlib import Path
import cv2

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.3

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / float(area1 + area2 - inter_area)

def submit_predictions(model_path, source_path, save_project, save_name):
    # Load YOLO model
    model = YOLO(model_path)

    # Predict
    results = model.predict(
        source=source_path,
        imgsz=1080,
        project=save_project,
        name=save_name,
        save_txt=True,
        save_conf=True,
        conf=0.001,
        iou=0.7,
        device=0,
        verbose=True
    )

    print(f"✅ Raw predictions saved at: {save_project}/{save_name}/labels")

    # Inpaint
    save_folder = Path(save_project) / save_name
    labels_folder = save_folder / "labels"
    inpainted_folder = save_folder / "test_inpainted"
    inpainted_folder.mkdir(parents=True, exist_ok=True)

    for label_file in Path(labels_folder).glob("*.txt"):
        img_name = label_file.stem + ".PNG"
        img_path = Path(source_path) / img_name
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"⚠️ Image not found: {img_path}")
            continue

        height, width, _ = img.shape

        boxes = []
        scores = []

        with open(label_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            cls, x_center, y_center, w, h, conf = map(float, parts)

            if conf < CONFIDENCE_THRESHOLD:
                continue

            x_center *= width
            y_center *= height
            w *= width
            h *= height

            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)

        keep = []
        while boxes:
            max_idx = scores.index(max(scores))
            keep.append((boxes[max_idx], scores[max_idx]))
            box_keep = boxes[max_idx]
            boxes.pop(max_idx)
            scores.pop(max_idx)

            new_boxes = []
            new_scores = []
            for b, s in zip(boxes, scores):
                if compute_iou(box_keep, b) < IOU_THRESHOLD:
                    new_boxes.append(b)
                    new_scores.append(s)
            boxes = new_boxes
            scores = new_scores

        for box, conf in keep:
            x1, y1, x2, y2 = box
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
            label = f"pole {conf:.2f}"
            cv2.putText(img, label, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        save_path = inpainted_folder / img_name
        cv2.imwrite(str(save_path), img)

    print(f"✅ Inpainted images saved to: {inpainted_folder}")
