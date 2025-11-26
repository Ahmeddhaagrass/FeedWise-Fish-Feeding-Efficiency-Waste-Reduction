import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import datetime


BASE_DATA_DIR = r'D:\uniiiii\fish_feeding2.v1i.coco'
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')
VAL_DIR = os.path.join(BASE_DATA_DIR, 'valid')
TRAIN_ANNOTATIONS = os.path.join(TRAIN_DIR, '_annotations.coco.json')
VAL_ANNOTATIONS = os.path.join(VAL_DIR, '_annotations.coco.json')

IMG_SIZE = 520
BATCH_SIZE = 16
MODEL_VARIANT = 'B0'
LEARNING_RATE = 0.0005
FINE_TUNE_LEARNING_RATE = 1e-5
EPOCHS = 90
EPOCHS_FINE_TUNE = 10
EARLY_STOPPING_PATIENCE = 5

SAVE_DIR = 'training_results_coco11111231'
MODEL_NAME = f'efficientnet{MODEL_VARIANT}_finetuned_coco'
BEST_MODEL_FILE = os.path.join(SAVE_DIR, f'{MODEL_NAME}_best.keras')
FINAL_MODEL_FILE = os.path.join(SAVE_DIR, f'{MODEL_NAME}_final.keras')
PLOT_ACC_FILE = os.path.join(SAVE_DIR, f'{MODEL_NAME}_accuracy.png')
PLOT_LOSS_FILE = os.path.join(SAVE_DIR, f'{MODEL_NAME}_loss.png')
LOG_DIR = os.path.join(SAVE_DIR, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'logs'), exist_ok=True)


def load_coco_annotations(json_path, img_dir):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    cat_id_to_index = {cat['id']: i for i, cat in enumerate(sorted(coco_data['categories'], key=lambda x: x['name']))}
    class_names_list = [cat['name'] for cat in sorted(coco_data['categories'], key=lambda x: x['name'])]
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

    image_paths, labels, processed_ids = [], [], set()
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in processed_ids: continue
        if image_id not in image_id_to_filename: continue
        category_id = ann['category_id']
        image_path = os.path.join(img_dir, image_id_to_filename[image_id])
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(cat_id_to_index[category_id])
            processed_ids.add(image_id)

    return image_paths, labels, class_names_list, len(class_names_list)

train_image_paths, train_labels, class_names, NUM_CLASSES = load_coco_annotations(TRAIN_ANNOTATIONS, TRAIN_DIR)
val_image_paths, val_labels, _, _ = load_coco_annotations(VAL_ANNOTATIONS, VAL_DIR)

CLASS_NAMES = class_names

train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_image_paths, val_labels))


def load_and_preprocess_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.cast(img, tf.float32) / 255.0, label

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)


EfficientNet = getattr(tf.keras.applications, f'EfficientNet{MODEL_VARIANT}')
base_model = EfficientNet(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)


optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_list = [
    ModelCheckpoint(BEST_MODEL_FILE, monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
    TensorBoard(log_dir=LOG_DIR)
]

print("Training with frozen base...")
history1 = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks_list
)


print("Unfreezing base model for fine-tuning...")
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_list.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1))

history2 = model.fit(
    train_ds,
    epochs=EPOCHS_FINE_TUNE,
    validation_data=val_ds,
    callbacks=callbacks_list
)


model.save(FINAL_MODEL_FILE)


acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

epochs_range = range(len(acc))

plt.figure()
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.savefig(PLOT_ACC_FILE)

plt.figure()
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.savefig(PLOT_LOSS_FILE)

print("Training complete.")
print(f"Best model saved to: {BEST_MODEL_FILE}")
print(f"Final model saved to: {FINAL_MODEL_FILE}")
