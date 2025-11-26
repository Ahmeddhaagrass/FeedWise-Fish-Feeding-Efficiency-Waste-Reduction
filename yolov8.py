from ultralytics import YOLO
from ultralytics.utils import LOGGER
import os
import argparse
import shutil
import time
import json
import pandas as pd


yaml_file = r'D:\fish_feeding2.v1i.yolov8/data.yaml'
val_image_folder = r"D:\fish_feeding2.v1i.yolov8/valid/images"
val_label_folder = r"D:\fish_feeding2.v1i.yolov8/valid/labels"
training_run_name = "yolov8_model_fish_feeding"
training_project_dir = "runs/detect"
weights_path = os.path.join(training_project_dir, training_run_name, 'weights/best.pt')
results_dir = os.path.join(training_project_dir, training_run_name, 'eval_results')
os.makedirs(results_dir, exist_ok=True)

def save_metrics(metrics_obj, filename):
    if not metrics_obj or not hasattr(metrics_obj, 'results_direct'):
        print(f"No metrics available to save for {filename}.")
        return
    results_path = os.path.join(results_dir, filename)
    try:
        with open(results_path, 'w') as f:
            json.dump(metrics_obj.results_dict, f, indent=4)
        print(f"Saved metrics to: {results_path}")
    except Exception as e:
        print(f"Failed to save metrics to {results_path}: {e}")

def run_inference_on_test_images(model):
    from pathlib import Path
    import cv2

    print("\nRunning inference on test images and saving predictions...")
    test_image_dir = None
    with open(yaml_file, 'r') as f:
        for line in f:
            if line.strip().startswith("test:"):
                test_image_dir = line.split(':', 1)[1].strip().strip('"')
                break

    if not test_image_dir or not os.path.isdir(test_image_dir):
        print("Test image directory not found or not defined in YAML.")
        return

    output_dir = os.path.join(results_dir, 'test_predictions')
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(test_image_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    records = []
    for img_file in image_files:
        img_path = os.path.join(test_image_dir, img_file)
        results = model.predict(source=img_path, save=False, conf=0.25, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    xyxy = box.xyxy[0].tolist()
                    records.append({
                        'image': img_file,
                        'class': model.names.get(cls, str(cls)),
                        'confidence': round(conf, 4),
                        'x1': round(xyxy[0], 2),
                        'y1': round(xyxy[1], 2),
                        'x2': round(xyxy[2], 2),
                        'y2': round(xyxy[3], 2)
                    })

        model.predict(source=img_path, save=True, conf=0.25, project=output_dir, name="", exist_ok=True, verbose=False)

    if records:
        df = pd.DataFrame(records)
        csv_path = os.path.join(results_dir, "test_predictions.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved test predictions CSV to: {csv_path}")
    else:
        print("No predictions were made on test images.")

def main(args):
    print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    model = load_or_tune_and_train_model(force_train=True)
    if model is None:
        print("\nERROR: Model loading or training failed. Exiting.")
        return

    print("\nRunning built-in YOLO validation...")
    try:
        metrics = model.val(data=yaml_file, split='val')
        print("Official YOLOv8 Validation Metrics:")
        if metrics and hasattr(metrics, 'results_dict'):
            print(f"  mAP50-95 (Box): {metrics.box.map:.4f}")
            print(f"  mAP50 (Box):   {metrics.box.map50:.4f}")
            print(f"  Precision (Box): {metrics.box.mp:.4f}")
            print(f"  Recall (Box):    {metrics.box.mr:.4f}")
        else:
            print("  Could not retrieve detailed metrics dictionary from model.val().")
        save_metrics(metrics, "val_metrics.json")
    except Exception as e:
        print(f"ERROR during built-in validation: {e}")

    print("\nRunning built-in YOLO test evaluation (if test split is defined)...")
    try:
        test_metrics = model.val(data=yaml_file, split='test')
        print("Test Set Evaluation Metrics:")
        if test_metrics and hasattr(test_metrics, 'results_dict'):
            print(f"  mAP50-95 (Box): {test_metrics.box.map:.4f}")
            print(f"  mAP50 (Box):   {test_metrics.box.map50:.4f}")
            print(f"  Precision (Box): {test_metrics.box.mp:.4f}")
            print(f"  Recall (Box):    {test_metrics.box.mr:.4f}")
        else:
            print("  Could not retrieve detailed metrics dictionary from model.val().")
        save_metrics(test_metrics, "test_metrics.json")
    except Exception as e:
        print(f"ERROR during test evaluation: {e}")

    run_inference_on_test_images(model)
    print(f"\nScript finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

def load_or_tune_and_train_model(force_train=False):
    global weights_path, yaml_file, training_run_name, training_project_dir

    if os.path.exists(weights_path) and not force_train:
        print(f"\nLoading existing best model from: {weights_path}")
        try:
            return YOLO(weights_path)
        except Exception as e:
            print(f"ERROR: Failed to load existing model: {e}")

    if os.path.exists(weights_path) and force_train:
        old_run_dir = os.path.join(training_project_dir, training_run_name)
        if os.path.exists(old_run_dir):
            print(f"Removing old run directory: {old_run_dir}")
            shutil.rmtree(old_run_dir)

    base_model_name = "yolov8m.pt"
    try:
        model = YOLO(base_model_name)
        print(f"Loaded pre-trained base model: {base_model_name}")
    except Exception as e:
        print(f"ERROR: Could not load base model: {e}")
        return None

    print("\nStarting hyperparameter tuning...")
    try:
        model.tune(
            data=yaml_file,
            epochs=100,
            iterations=15,
            imgsz=640,
            val=True,
            project=training_project_dir,
            name=training_run_name + "_tuned",
            exist_ok=True
        )
        print("Hyperparameter tuning complete.")
    except Exception as e:
        print(f"ERROR during tuning: {e}")

    print("\nStarting final model training with best hyperparameters...")
    try:
        best_hyp_path = os.path.join(training_project_dir, training_run_name + "_tuned", "hyp.yaml")
        model.train(
            data=yaml_file,
            epochs=100,
            batch=16,
            imgsz=640,
            name=training_run_name,
            project=training_project_dir,
            exist_ok=True,
            val=True,
            hyp=best_hyp_path
        )
        print("Training complete!")

        if os.path.exists(weights_path):
            return YOLO(weights_path)
        else:
            last_weights_path = os.path.join(training_project_dir, training_run_name, 'weights/last.pt')
            if os.path.exists(last_weights_path):
                print(f"best.pt not found, using last.pt")
                return YOLO(last_weights_path)
    except Exception as e:
        print(f"ERROR during training: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a YOLOv8 object detection model with tuning.')
    parser.add_argument('--force-train', action='store_true', help='Force retraining the model')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        LOGGER.error(f"Critical error during script execution:", exc_info=True)
        print(f"\nCRITICAL ERROR: {e}")