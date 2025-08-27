import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from collections import Counter
import requests
from io import BytesIO
from sklearn.utils.class_weight import compute_class_weight

LABELS_CSV_URL = "https://example.com/labels.csv"
IMG_SIZE = (32, 32)
BATCH_SIZE = 16
EPOCHS = 100
VALIDATION_SPLIT = 0.2
MODEL_SAVE_PATH = "captcha_final.h5"
ENSEMBLE_MODELS = 3

def load_labels(url):
    print("downloading labels...")
    response = requests.get(url)
    df = pd.read_csv(BytesIO(response.content))
    print(f"loaded {len(df)} labels")
    return df

def extract_rois(img_path, expected_digits=4):
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
        
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_cl = clahe.apply(gray)
    
    thresholds = [170, 150, 190, 130, 210, 110]
    best_rois = []
    best_score = 0
    
    for thresh_val in thresholds:
        _, th = cv2.threshold(gray_cl, thresh_val, 255, cv2.THRESH_BINARY)
        
        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k2)
        
        scale = 3
        th_up = cv2.resize(th, (th.shape[1]*scale, th.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
        
        contours, _ = cv2.findContours(th_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area < 80: continue
            if h < th_up.shape[0]*0.12: continue
            if w < 8: continue
            
            aspect_ratio = w / h
            if aspect_ratio > 2.0 or aspect_ratio < 0.3: continue
            
            boxes.append((x,y,w,h,area))
        
        if len(boxes) >= expected_digits:
            boxes = sorted(boxes, key=lambda b: b[0])[:expected_digits]
            
            if len(boxes) > 1:
                spacings = [boxes[i+1][0] - (boxes[i][0] + boxes[i][2]) for i in range(len(boxes)-1)]
                spacing_std = np.std(spacings) if len(spacings) > 1 else 0
                size_consistency = np.std([b[3] for b in boxes])
                score = 1000 / (1 + spacing_std + size_consistency)
            else:
                score = 100
            
            if score > best_score:
                best_score = score
                rois = []
                for i,(x,y,w,h,_) in enumerate(boxes):
                    pad_x = max(2, int(w*0.15))
                    pad_y = max(2, int(h*0.15))
                    x1 = max(0, x-pad_x); y1 = max(0, y-pad_y)
                    x2 = min(th_up.shape[1], x+w+pad_x); y2 = min(th_up.shape[0], y+h+pad_y)
                    roi = th_up[y1:y2, x1:x2]
                    rois.append(roi)
                
                if len(rois) == expected_digits:
                    best_rois = rois
    
    return best_rois

def augment_roi(roi):
    augmented_rois = [roi]
    
    for angle in [-5, 5, -3, 3]:
        center = (roi.shape[1]//2, roi.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]))
        augmented_rois.append(rotated)
    
    for scale in [0.9, 1.1]:
        h, w = roi.shape
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(roi, (new_w, new_h))
        
        if scale < 1.0:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            scaled = cv2.copyMakeBorder(scaled, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, cv2.BORDER_CONSTANT, value=255)
        else:
            crop_h = (new_h - h) // 2
            crop_w = (new_w - w) // 2
            scaled = scaled[crop_h:crop_h+h, crop_w:crop_w+w]
        
        augmented_rois.append(scaled)
    
    for noise_level in [5, 10]:
        noisy = roi.copy().astype(np.float32)
        noise = np.random.normal(0, noise_level, roi.shape)
        noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
        augmented_rois.append(noisy)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    eroded = cv2.erode(roi, kernel, iterations=1)
    dilated = cv2.dilate(roi, kernel, iterations=1)
    augmented_rois.extend([eroded, dilated])
    
    return augmented_rois

def load_image(filename):
    possible_paths = [
        f"attempts/captcha_images/{filename}",
        f"captcha_images/{filename}",
        filename
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return cv2.imread(path)
    
    print(f"image not found: {filename}")
    return None

def prepare_data(labels_df, use_augmentation=True):
    X = []
    y = []
    
    print("processing images...")
    
    for idx, row in labels_df.iterrows():
        filename = row['filename']
        predicted = str(row['predicted'])
        
        img = load_image(filename)
        if img is None:
            continue
            
        rois = extract_rois(img, expected_digits=len(predicted))
        
        if len(rois) != len(predicted):
            print(f"skip {filename}: {len(rois)} rois vs {len(predicted)} digits")
            continue
            
        for roi, digit_char in zip(rois, predicted):
            if digit_char.isdigit():
                if use_augmentation:
                    augmented_rois = augment_roi(roi)
                    for aug_roi in augmented_rois:
                        roi_resized = cv2.resize(aug_roi, IMG_SIZE)
                        roi_normalized = roi_resized.astype('float32') / 255.0
                        X.append(roi_normalized)
                        y.append(int(digit_char))
                else:
                    roi_resized = cv2.resize(roi, IMG_SIZE)
                    roi_normalized = roi_resized.astype('float32') / 255.0
                    X.append(roi_normalized)
                    y.append(int(digit_char))
        
        if (idx + 1) % 50 == 0:
            print(f"processed {idx + 1}/{len(labels_df)} images")
    
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], IMG_SIZE[0], IMG_SIZE[1], 1)
    
    print(f"data prepared: {X.shape[0]} samples")
    print(f"class distribution: {Counter(y)}")
    
    return X, y

def create_model():
    input_layer = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    residual = x
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    if residual.shape[-1] != x.shape[-1]:
        residual = Conv2D(64, (1, 1), padding='same')(residual)
    x = Add()([x, residual])
    
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_ensemble(X_train, y_train, X_val, y_val, class_weights):
    models = []
    histories = []
    
    for i in range(ENSEMBLE_MODELS):
        print(f"training model {i+1}/{ENSEMBLE_MODELS}")
        
        model = create_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        tf.random.set_seed(42 + i)
        
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        models.append(model)
        histories.append(history)
        
        model.save(f"captcha_model_{i+1}.h5")
    
    return models, histories

def main():
    print("captcha model training")
    
    labels_df = load_labels(LABELS_CSV_URL)
    X, y = prepare_data(labels_df, use_augmentation=True)
    
    if len(X) == 0:
        print("no training data found")
        return
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"class weights: {class_weight_dict}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )
    
    models, histories = train_ensemble(X_train, y_train, X_val, y_val, class_weight_dict)
    
    best_model = None
    best_accuracy = 0
    
    for i, model in enumerate(models):
        val_accuracy = max(histories[i].history['val_accuracy'])
        print(f"model {i+1} validation accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
    
    if best_model:
        best_model.save(MODEL_SAVE_PATH)
        print(f"best model saved: {MODEL_SAVE_PATH}")
    
    print("training complete")

if __name__ == "__main__":
    main()
