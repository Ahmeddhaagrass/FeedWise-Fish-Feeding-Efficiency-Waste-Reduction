import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import pandas as pd
import numpy as np
import traceback
from collections import Counter


def load_annotations(annotations_file):
    try:
        with open(annotations_file) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotations file not found at {annotations_file}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {annotations_file}")
        return None

def map_classes(annotations):
    if not annotations or 'categories' not in annotations:
        print("Error: Invalid annotations format or missing 'categories'.")
        return {}
    return {category['id']: category['name'] for category in annotations['categories']}

def create_class_to_idx(classes):

    if not classes:
        return {}
    class_names_sorted = sorted(list(classes.values()))
    return {name: i for i, name in enumerate(class_names_sorted)}

def get_class_name_from_id(annotations, cat_id):
    if not annotations or 'categories' not in annotations:
        return None

    for category in annotations['categories']:
        if category['id'] == cat_id:
            return category['name']
    return None

def prepare_img_info(annotations, img_dir, class_to_idx):
    img_info = []
    if not annotations or 'images' not in annotations or 'annotations' not in annotations or not class_to_idx:
         print("Warning: Invalid annotations, missing image data, or empty class_to_idx. Cannot prepare image info.")
         return []


    image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}

    missing_files_count = 0
    skipped_annotations = 0

    for ann in annotations['annotations']:
        img_id = ann['image_id']
        img_filename = image_id_to_filename.get(img_id)
        cat_id = ann['category_id']
        class_name = get_class_name_from_id(annotations, cat_id)

        if img_filename and class_name and class_name in class_to_idx:
            img_path = os.path.join(img_dir, img_filename)

            if os.path.exists(img_path):
                label = class_to_idx[class_name]
                img_info.append({'img_path': img_path, 'label': label})
            else:

                missing_files_count += 1
                skipped_annotations += 1
        else:
                skipped_annotations += 1

    if missing_files_count > 0:
        print(f"Warning: {missing_files_count} image files listed in annotations were not found on disk.")

    return img_info


class CocoDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = load_annotations(annotations_file)
        self.img_dir = img_dir
        self.transform = transform


        if self.annotations is None:
            print(f"Error: Failed to load annotations from {annotations_file}. Dataset will be empty.")
            self.classes = {}
            self.class_to_idx = {}
            self.idx_to_class = {}
            self.img_info = []
            return

        self.classes = map_classes(self.annotations)
        self.class_to_idx = create_class_to_idx(self.classes)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        if not self.classes or not self.class_to_idx:
             print(f"Warning: No classes found or mapped for {annotations_file}. Dataset might be empty or incorrectly configured.")
             self.img_info = []
        else:
            self.img_info = prepare_img_info(self.annotations, img_dir, self.class_to_idx)
            print(f"Loaded {len(self.img_info)} images from {annotations_file}")
            if self.classes:
                 print(f"Found {len(self.classes)} classes: {list(self.classes.values())}")
            if len(self.img_info) == 0:
                 print(f"Warning: No images loaded. Check image directory ('{img_dir}') and annotations file ('{annotations_file}').")

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):

        if not self.img_info:
            raise IndexError("Dataset image info is empty.")
        if idx >= len(self.img_info):
             raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.img_info)}.")

        img_path = self.img_info[idx]['img_path']
        label = self.img_info[idx]['label']

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image not found during open (should have been checked earlier): {img_path}")
            return None, None
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)

        return image, label


IMG_SIZE = (224, 224)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])


val_test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])


def load_data(batch_size=32, num_workers=0):

    base_path = r'D:\uniiiii\fish_feeding2.v1i.coco'
    train_ann_file = os.path.join(base_path, 'train/_annotations.coco.json')
    train_img_dir = os.path.join(base_path, 'train')
    val_ann_file = os.path.join(base_path, 'valid/_annotations.coco.json')
    val_img_dir = os.path.join(base_path, 'valid')
    test_ann_file = os.path.join(base_path, 'test/_annotations.coco.json')
    test_img_dir = os.path.join(base_path, 'test')


    if not os.path.isdir(base_path):
        print(f"Error: Base directory not found: {base_path}")
        print("Please adjust the 'base_path' variable in the 'load_data' function.")
        return None, None, None, None, None, None
    train_dataset = CocoDataset(annotations_file=train_ann_file,
                                img_dir=train_img_dir, transform=transform)
    val_dataset = CocoDataset(annotations_file=val_ann_file,
                                img_dir=val_img_dir, transform=val_test_transform)
    test_dataset = CocoDataset(annotations_file=test_ann_file,
                                 img_dir=test_img_dir, transform=val_test_transform)





    def collate_fn_skip_error(batch):
        batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
        if not batch: return None

        try:
             return torch.utils.data.dataloader.default_collate(batch)
        except Exception as e:
             print(f"Error during collate_fn: {e}. Skipping batch.")
             traceback.print_exc()
             return None


    pin_memory = torch.cuda.is_available() and num_workers > 0


    print(f"Using num_workers = {num_workers} for DataLoaders. pin_memory = {pin_memory}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn_skip_error, pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_skip_error, pin_memory=pin_memory) if len(val_dataset) > 0 else None

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_skip_error, pin_memory=pin_memory) if len(test_dataset) > 0 else None

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx, train_dataset.idx_to_class, train_dataset.classes


def initialize_model(model_arch, num_classes, pretrained=True):

    model = None
    weights = models.get_model_weights(model_arch).DEFAULT if pretrained else None

    if model_arch == 'resnet18':
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_arch == 'resnet34':
        model = models.resnet34(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_arch}")

    print(f"Initialized model: {model_arch} with {num_classes} classes. Pretrained={pretrained}")
    return model

def format_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    train_loss_history, val_loss_history, train_acc_history, val_acc_history = [], [], [], []
    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print("-" * 20)
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 20)
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0


        pbar_train = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar_train:
            if batch is None:
                print("Warning: Skipping an empty batch in training (likely due to previous image errors).")
                continue
            try:
                images, labels = batch
                if images is None or labels is None:
                   print("Warning: Skipping batch with None image/label after collate.")
                   continue
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_samples += labels.size(0)


                current_avg_loss = running_loss / total_samples if total_samples > 0 else 0
                current_avg_acc = correct_preds / total_samples if total_samples > 0 else 0
                pbar_train.set_postfix(loss=f"{current_avg_loss:.4f}", acc=f"{current_avg_acc:.4f}")

            except Exception as e:
                print(f"Error during training batch: {e}")
                print("Skipping batch.")
                traceback.print_exc()
                continue


        epoch_train_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_train_acc = correct_preds / total_samples if total_samples > 0 else 0
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)

        epoch_val_loss = float('nan')
        epoch_val_acc = float('nan')
        if val_loader and len(val_loader.dataset) > 0:
            model.eval()
            running_val_loss = 0.0
            correct_val_preds = 0
            total_val_samples = 0

            pbar_val = tqdm(val_loader, desc="Validation", leave=False)
            with torch.no_grad():
                for batch in pbar_val:
                    if batch is None:

                        continue

                    try:
                        images, labels = batch
                        if images is None or labels is None:

                           continue

                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                        if torch.isnan(loss):
                           print(f"WARNING: NaN validation loss detected at Epoch {epoch+1}.")

                           running_val_loss = float('nan')
                           break

                        running_val_loss += loss.item() * images.size(0)
                        _, predicted = torch.max(outputs, 1)
                        correct_val_preds += (predicted == labels).sum().item()
                        total_val_samples += labels.size(0)

                        current_avg_val_loss = running_val_loss / total_val_samples if total_val_samples > 0 else 0
                        current_avg_val_acc = correct_val_preds / total_val_samples if total_val_samples > 0 else 0
                        pbar_val.set_postfix(loss=f"{current_avg_val_loss:.4f}", acc=f"{current_avg_val_acc:.4f}")

                    except Exception as e:
                        print(f"Error during validation batch: {e}")
                        print("Skipping batch.")
                        traceback.print_exc()
                        continue

            epoch_val_loss = running_val_loss / total_val_samples if total_val_samples > 0 and not np.isnan(running_val_loss) else float('nan')
            epoch_val_acc = correct_val_preds / total_val_samples if total_val_samples > 0 else 0.0

        else:
            print("Skipping validation phase as validation loader is not available or empty.")
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)
        epoch_duration = time.time() - epoch_start_time
        total_elapsed_time = time.time() - total_start_time
        avg_epoch_time = total_elapsed_time / (epoch + 1)
        estimated_remaining_time = avg_epoch_time * (epochs - (epoch + 1))

        print(f"Epoch Summary [{epoch+1}/{epochs}]:")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        val_loss_str = f"{epoch_val_loss:.4f}" if not np.isnan(epoch_val_loss) else "N/A (No Val Data or NaN)"
        val_acc_str = f"{epoch_val_acc:.4f}" if not np.isnan(epoch_val_acc) else "N/A"
        print(f"  Val Loss  : {val_loss_str}, Val Acc  : {val_acc_str}")
        print(f"  Epoch Time: {format_time(epoch_duration)}")
        print(f"  Est. Remaining: {format_time(estimated_remaining_time)}")
        print("-" * 20)

    total_training_time = time.time() - total_start_time
    print(f"\nTraining Complete. Total time: {format_time(total_training_time)}")
    return train_loss_history, val_loss_history, train_acc_history, val_acc_history


def plot_metrics(train_loss, val_loss, train_acc, val_acc, epochs, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Train Loss', marker='o')

    if not all(np.isnan(vl) for vl in val_loss):
        plt.plot(epochs_range, val_loss, label='Validation Loss', marker='x')
    else:
        print("Skipping validation loss plot (No valid data).")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epochs')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label='Train Accuracy', marker='o')
    if not all(np.isnan(va) for va in val_acc):
        plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='x')
    else:
        print("Skipping validation accuracy plot (No valid data).")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Epochs')
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    try:
        plt.savefig(plot_path)
        print(f"\nSaved training metrics plot to {plot_path}")
    except Exception as e:
        print(f"Error saving training metrics plot: {e}")
    finally:
        plt.close()
def evaluate_and_save_metrics(model, test_loader, idx_to_class, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    y_true, y_pred = [], []
    total_samples = 0
    correct_preds = 0
    print("\n Detailed Test Set Evaluation ")
    pbar_test = tqdm(test_loader, desc="Evaluating Test Set")
    with torch.no_grad():
        for batch in pbar_test:

            if batch is None:
                print("Warning: Skipping an empty batch during final evaluation.")
                continue
            try:
                images, labels = batch
                if images is None or labels is None:
                    print("Warning: Skipping batch with None image/label after collate (evaluation).")
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                correct_preds += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                current_acc = correct_preds / total_samples if total_samples > 0 else 0
                pbar_test.set_postfix(acc=f"{current_acc:.4f}")

            except Exception as e:
                print(f"Error during evaluation batch: {e}")
                print("Skipping batch.")
                traceback.print_exc()
                continue
    if total_samples == 0:
        print("Error: No samples were processed during evaluation. Cannot generate metrics.")
        return None
    test_acc = correct_preds / total_samples if total_samples > 0 else 0.0
    print(f"\nFinal Test Accuracy: {test_acc:.4f} ({correct_preds}/{total_samples})")
    class_names = [idx_to_class.get(i, f"Unknown_Index_{i}") for i in range(len(idx_to_class))]


    try:
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))


        plt.figure(figsize=(max(8, len(class_names)*0.8), max(6, len(class_names)*0.6)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"\nSaved confusion matrix plot to {cm_path}")
    except Exception as e:
        print(f"Error generating/saving/printing confusion matrix: {e}")
        traceback.print_exc()



    try:
        report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        report_dict['accuracy'] = {'precision': None, 'recall': None, 'f1-score': test_acc, 'support': total_samples}
        df_report = pd.DataFrame(report_dict).transpose()
        df_report_formatted = df_report.applymap(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else x)
        df_report_formatted['support'] = df_report['support'].astype(float).astype(int)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
             print(df_report_formatted)

        report_path_excel = os.path.join(output_dir, 'classification_report.xlsx')
        report_path_csv = os.path.join(output_dir, 'classification_report.csv')
        df_report_formatted.to_excel(report_path_excel)
        df_report_formatted.to_csv(report_path_csv)
        print(f"\nSaved classification report to {report_path_excel} and {report_path_csv}")

        metrics_to_plot = ['precision', 'recall', 'f1-score']
        report_df_for_plot = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        for metric in metrics_to_plot:
            if metric in report_df_for_plot.columns:
                try:
                    metric_values = pd.to_numeric(report_df_for_plot[metric], errors='coerce').fillna(0)
                    plt.figure(figsize=(max(8, len(class_names)*0.6), 5))
                    plt.bar(report_df_for_plot.index, metric_values, color='skyblue')
                    plt.title(f'{metric.replace("-", " ").capitalize()} per Class')
                    plt.ylabel(metric.replace("-", " ").capitalize())
                    plt.xlabel('Class')
                    plt.xticks(rotation=45, ha='right')
                    plt.ylim(0, 1.1)
                    plt.grid(axis='y', linestyle='--')
                    plt.tight_layout()
                    metric_plot_path = os.path.join(output_dir, f'{metric}_per_class.png')
                    plt.savefig(metric_plot_path)

                except Exception as plot_err:
                     print(f"Error generating/saving plot for metric '{metric}': {plot_err}")
                finally:
                     plt.close()

    except Exception as e:
        print(f"Error generating/saving/printing classification report or metric plots: {e}")
        traceback.print_exc()

    return test_acc

def compute_class_weights(dataset, num_classes):


    if not dataset or not hasattr(dataset, 'img_info') or not dataset.img_info:
        print("Warning: Cannot compute class weights. Dataset or img_info is empty/invalid.")

        return torch.ones(num_classes)

    labels = [info['label'] for info in dataset.img_info if info and 'label' in info]

    if not labels:
        print("Warning: No labels found in dataset to compute class weights.")
        return torch.ones(num_classes)

    label_counts = Counter(labels)
    total_samples = len(labels)

    class_weights = [0.0] * num_classes


    for i in range(num_classes):
        count = label_counts.get(i, 0)

        effective_count = max(count, 1)
        class_weights[i] = total_samples / (num_classes * effective_count)

        print(f"Computed Class Counts: {dict(label_counts)}")
    return torch.FloatTensor(class_weights)


def main():





    EPOCHS = 20
    BATCH_SIZE = 128

    LEARNING_RATE =0.005

    OPTIMIZER_TYPE = 'Adam'
    LOSS_FUNCTION = 'CrossEntropyLoss'
    MODEL_ARCH = 'resnet34'
    USE_PRETRAINED = True
    NUM_WORKERS = 0
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    OUTPUT_DIR = 'training_results_FAST_PRINT22222'



    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results will be saved in: {os.path.abspath(OUTPUT_DIR)}")
    print("--- RUNNING WITH HYPERPARAMETERS OPTIMIZED FOR SPEED ---")
    print(f"--- WARNING: Accuracy may be significantly reduced ---")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nSelected device: {device}")

    try:

        train_loader, val_loader, test_loader, class_to_idx, idx_to_class, classes_dict = load_data(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )


        if train_loader is None or not class_to_idx or not idx_to_class:
             print("Error: Essential data components (train_loader, class_to_idx, idx_to_class) failed to load. Exiting.")
             return


        num_classes = len(class_to_idx)
        print(f"Number of classes found: {num_classes}")
        if num_classes == 0:
            print("Error: No classes detected. Cannot initialize model. Exiting.")
            return


        print("Computing class weights...")

        if train_loader and hasattr(train_loader, 'dataset'):
             class_weights = compute_class_weights(train_loader.dataset, num_classes)
        else:
             print("Warning: Could not compute class weights because train_loader or dataset is invalid.")
             class_weights = torch.ones(num_classes)

        print(f"Class Weights: {class_weights}")
        class_weights = class_weights.to(device)


        if LOSS_FUNCTION == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using {LOSS_FUNCTION} with calculated class weights.")
        else:

            print(f"Warning: Unsupported LOSS_FUNCTION '{LOSS_FUNCTION}'. Defaulting to CrossEntropyLoss without weights.")
            criterion = nn.CrossEntropyLoss()


    except Exception as e:
        print(f"Critical error during data loading or weight calculation setup: {e}")
        traceback.print_exc()
        return


    config_data = {
        'model_architecture': MODEL_ARCH,
        'use_pretrained': USE_PRETRAINED,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'optimizer': OPTIMIZER_TYPE,
        'loss_function': LOSS_FUNCTION,
        'class_weighting_enabled': LOSS_FUNCTION == 'CrossEntropyLoss',
        'num_workers': NUM_WORKERS,
        'output_directory': OUTPUT_DIR,
        'num_classes': num_classes,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'classes_by_original_id': classes_dict
    }
    config_path = os.path.join(OUTPUT_DIR, 'config_and_class_map.json')
    try:
        with open(config_path, 'w') as f:

            def default_serializer(obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, torch.Tensor): return obj.tolist()
                if isinstance(obj, (set, tuple)): return list(obj)

                try:
                    return str(obj)
                except Exception:
                     return f"Unserializable Type: {type(obj)}"

            json.dump(config_data, f, indent=4, default=default_serializer)
        print(f"Saved configuration and class mappings to {config_path}")
    except TypeError as e:
        print(f"Error saving configuration: Data might not be JSON serializable - {e}")
    except Exception as e:
        print(f"Error saving configuration: {e}")
        traceback.print_exc()



    print("\nInitializing model...")
    try:
        model = initialize_model(MODEL_ARCH, num_classes, pretrained=USE_PRETRAINED)
    except Exception as e:
        print(f"Error initializing model: {e}")
        traceback.print_exc()
        return

    try:
        if OPTIMIZER_TYPE == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        elif OPTIMIZER_TYPE == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        else:
            print(f"Warning: Unsupported OPTIMIZER_TYPE '{OPTIMIZER_TYPE}'. Defaulting to Adam.")
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        print(f"Using Optimizer: {type(optimizer).__name__} with LR: {LEARNING_RATE}")
    except Exception as e:
        print(f"Error initializing optimizer: {e}")
        traceback.print_exc()
        return



    print("\nStarting training...")
    try:
        train_loss, val_loss, train_acc, val_acc = train_model(
            model, criterion, optimizer, train_loader, val_loader, epochs=EPOCHS

        )

        if train_loss is None:
             print("Training was interrupted due to an error (e.g., NaN loss). Skipping post-training steps.")
             return

    except Exception as e:
        print(f"Critical error during training loop: {e}")
        traceback.print_exc()
        return


    history_data = {
        'train_loss': train_loss,
        'val_loss': [vl if not np.isnan(vl) else None for vl in val_loss],
        'train_acc': train_acc,
        'val_acc': [va if not np.isnan(va) else None for va in val_acc]
    }
    history_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    try:
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=4)
        print(f"Saved training history to {history_path}")
    except Exception as e:
        print(f"Error saving training history: {e}")
        traceback.print_exc()



    print("\nPlotting training metrics...")

    if train_loss and train_acc:
        plot_metrics(train_loss, val_loss, train_acc, val_acc, epochs=EPOCHS, output_dir=OUTPUT_DIR)
    else:
        print("Skipping plotting metrics due to invalid training history.")



    print("\nEvaluating final model on test set and saving/printing metrics...")
    if test_loader and len(test_loader.dataset) > 0:
        try:

            test_accuracy = evaluate_and_save_metrics(model, test_loader, idx_to_class, output_dir=OUTPUT_DIR)
            if test_accuracy is not None:

                print(f"Detailed evaluation complete (Metrics printed above and saved).")
            else:
                print("Evaluation could not be completed (returned None).")
        except Exception as e:
             print(f"Error during final evaluation: {e}")
             traceback.print_exc()
    else:
        print("Skipping final evaluation because the test loader is empty or invalid.")
        test_accuracy = None

    if train_loss is not None:
        model_save_path = os.path.join(OUTPUT_DIR, f'{MODEL_ARCH}_final_e{EPOCHS}_fast.pth')
        print(f"\nSaving final trained model to {model_save_path}...")
        try:
            torch.save(model.state_dict(), model_save_path)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving model: {e}")
            traceback.print_exc()
    else:
        print("\nSkipping final model saving because training did not complete successfully.")

    print("\n--- Script Finished (Optimized for Speed, Printing Metrics) ---")


if __name__ == '__main__':

    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s). Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Training will run on CPU (which will be slower).")

    main()