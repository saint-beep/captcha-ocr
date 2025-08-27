import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pytesseract
from PIL import Image

class CaptchaPredictor:
    def __init__(self, model_path='captcha_final.h5'):
        self.model = None
        if os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                print(f"model loaded: {model_path}")
            except Exception as e:
                print(f"error loading model: {e}")
        
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, []
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_versions = []
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced1 = clahe.apply(gray)
        blurred1 = cv2.GaussianBlur(enhanced1, (3, 3), 0)
        processed_versions.append(blurred1)
        
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        processed_versions.append(morph)
        
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        processed_versions.append(adaptive)
        
        return gray, processed_versions
    
    def extract_digits(self, image):
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digit_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 15 <= w <= 60 and 20 <= h <= 80:
                digit_contours.append((x, y, w, h))
        
        digit_contours.sort(key=lambda x: x[0])
        
        digit_images = []
        for x, y, w, h in digit_contours:
            padding = 5
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w + padding)
            y_end = min(image.shape[0], y + h + padding)
            
            digit_roi = image[y_start:y_end, x_start:x_end]
            
            if digit_roi.size > 0:
                digit_resized = cv2.resize(digit_roi, (28, 28))
                digit_images.append(digit_resized)
        
        return digit_images
    
    def predict_captcha(self, image_path):
        if self.model is None:
            return self.fallback_ocr(image_path)
        
        gray, processed_versions = self.preprocess_image(image_path)
        if gray is None:
            return "", 0.0
        
        best_prediction = ""
        best_confidence = 0.0
        
        for processed_img in processed_versions:
            digit_images = self.extract_digits(processed_img)
            
            if len(digit_images) == 4:
                prediction, confidence = self.predict_digits(digit_images)
                
                if confidence > best_confidence:
                    best_prediction = prediction
                    best_confidence = confidence
        
        if best_confidence < 0.3:
            ocr_result = self.fallback_ocr(image_path)
            if ocr_result[1] > best_confidence:
                return ocr_result
        
        return best_prediction, best_confidence
    
    def predict_digits(self, digit_images):
        if len(digit_images) != 4:
            return "", 0.0
        
        predictions = []
        confidences = []
        
        for digit_img in digit_images:
            digit_normalized = digit_img.astype('float32') / 255.0
            digit_input = digit_normalized.reshape(1, 28, 28, 1)
            
            pred = self.model.predict(digit_input, verbose=0)
            predicted_digit = np.argmax(pred[0])
            confidence = np.max(pred[0])
            
            predictions.append(str(predicted_digit))
            confidences.append(confidence)
        
        avg_confidence = np.mean(confidences)
        result = ''.join(predictions)
        
        return result, avg_confidence
    
    def fallback_ocr(self, image_path):
        try:
            img = Image.open(image_path)
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(img, config=custom_config).strip()
            
            if len(text) == 4 and text.isdigit():
                return text, 0.5
            else:
                return "", 0.0
        except Exception as e:
            print(f"ocr error: {e}")
            return "", 0.0

def test_predictor():
    predictor = CaptchaPredictor()
    
    test_folder = "attempts/captcha_images"
    if not os.path.exists(test_folder):
        print(f"folder {test_folder} does not exist")
        return
    
    image_files = [f for f in os.listdir(test_folder) if f.endswith('.png')]
    
    if not image_files:
        print("no png images found")
        return
    
    print(f"testing {len(image_files)} images...")
    
    complete_predictions = 0
    high_confidence = 0
    
    for img_file in image_files[:15]:
        img_path = os.path.join(test_folder, img_file)
        prediction, confidence = predictor.predict_captcha(img_path)
        
        if prediction:
            complete_predictions += 1
            if confidence > 0.7:
                high_confidence += 1
                status = "excellent"
            elif confidence > 0.5:
                status = "good"
            else:
                status = "low"
        else:
            status = "incomplete"
        
        print(f"{img_file}: {prediction:>4} (conf: {confidence:.3f}) {status}")
    
    print(f"complete: {complete_predictions}/{len(image_files[:15])} ({complete_predictions/15*100:.1f}%)")
    print(f"high confidence: {high_confidence}/{len(image_files[:15])} ({high_confidence/15*100:.1f}%)")

if __name__ == "__main__":
    test_predictor()
