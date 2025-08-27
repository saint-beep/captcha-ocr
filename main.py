import os, time, glob, csv, hashlib
import cv2
import numpy as np
from PIL import Image
import pytesseract
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

try:
    from tensorflow.keras.models import load_model
    tf_available = True
except Exception:
    tf_available = False

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
USERNAME = "your_username"
PASSWORD = "your_password"
URL = "https://example-site.com/"

EXPECTED_LENGTH = 4
MAX_RETRIES = 4
TOTAL_SESSIONS = 300
SUBMIT_WAIT = 1.2
DATA_FOLDER = "attempts"
MODEL_FILE = "captcha_final.h5"
AUTO_SAVE_SUCCESS = True
CAPTCHA_REFRESH_TIMEOUT = 6.0
TYPING_DELAY = 0.10
MAX_TYPING_ATTEMPTS = 3

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(os.path.join(DATA_FOLDER, "captcha_images"), exist_ok=True)
os.makedirs(os.path.join(DATA_FOLDER, "rois"), exist_ok=True)
os.makedirs(os.path.join(DATA_FOLDER, "labeled"), exist_ok=True)
os.makedirs(os.path.join(DATA_FOLDER, "unlabeled"), exist_ok=True)

LOG_FILE = os.path.join(DATA_FOLDER, "attempts_log.csv")
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "captcha_file", "sent_string", "method", "conf_score", "success", "notes"])

def get_file_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def extract_text_with_tesseract(pil_img, config):
    configs = [
        config,
        "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789",
        "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789", 
        "--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789",
        "--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789"
    ]
    
    best_result = ""
    best_confidence = 0
    
    for cfg in configs:
        try:
            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config=cfg)
            texts = []
            confidences = []
            n = len(data.get('text', []))
            for i in range(n):
                raw_text = data['text'][i]
                if raw_text is None: continue
                text = str(raw_text).strip()
                raw_conf = data['conf'][i]
                try:
                    conf = int(float(raw_conf))
                except Exception:
                    conf = -1
                if text and any(ch.isdigit() for ch in text):
                    digits = ''.join(filter(str.isdigit, text))
                    if digits:
                        texts.append(digits)
                        if conf >= 0:
                            confidences.append(conf)
            
            result_text = ''.join(texts)
            avg_conf = (sum(confidences)/len(confidences)) if confidences else 0
            
            if len(result_text) == EXPECTED_LENGTH and avg_conf > best_confidence:
                best_result = result_text
                best_confidence = avg_conf
            elif len(result_text) > len(best_result) and not best_result:
                best_result = result_text
                best_confidence = avg_conf
                
        except Exception:
            continue
    
    return best_result, best_confidence

def extract_digit_regions(img_path, debug_prefix=None):
    img = cv2.imread(img_path)
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    
    thresholds = [170, 150, 190, 130]
    best_regions = []
    
    for thresh_val in thresholds:
        _, binary = cv2.threshold(gray_enhanced, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        scale = 3
        binary_scaled = cv2.resize(binary, (binary.shape[1]*scale, binary.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
        
        if debug_prefix:
            cv2.imwrite(f"{debug_prefix}_mask_{thresh_val}.png", binary_scaled)
        
        contours, _ = cv2.findContours(binary_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if cv2.contourArea(c) < 80: continue
            if h < binary_scaled.shape[0]*0.12: continue
            if w < 8: continue
            boxes.append((x,y,w,h))
        
        if len(boxes) >= EXPECTED_LENGTH:
            boxes = sorted(boxes, key=lambda b: b[0])[:EXPECTED_LENGTH]
            regions = []
            for i,(x,y,w,h) in enumerate(boxes):
                pad_x = max(2, int(w*0.15))
                pad_y = max(2, int(h*0.15))
                x1 = max(0, x-pad_x); y1 = max(0, y-pad_y)
                x2 = min(binary_scaled.shape[1], x+w+pad_x); y2 = min(binary_scaled.shape[0], y+h+pad_y)
                roi = binary_scaled[y1:y2, x1:x2]
                regions.append(roi)
                if debug_prefix:
                    cv2.imwrite(f"{debug_prefix}_roi_{thresh_val}_{i}.png", roi)
            
            if len(regions) == EXPECTED_LENGTH:
                return regions
            elif len(regions) > len(best_regions):
                best_regions = regions
    
    return best_regions

def save_labeled_data(regions, label_string):
    saved = 0
    if len(regions) != len(label_string):
        filename = f"mismatch_{int(time.time()*1000)}.png"
        if regions:
            cv2.imwrite(os.path.join(DATA_FOLDER, "labeled", filename), cv2.resize(regions[0], (100,100)))
        else:
            cv2.imwrite(os.path.join(DATA_FOLDER, "labeled", filename), np.zeros((28,28), np.uint8))
        return saved
    for i, roi in enumerate(regions):
        label = label_string[i]
        label_dir = os.path.join(DATA_FOLDER, "labeled", label)
        os.makedirs(label_dir, exist_ok=True)
        output_path = os.path.join(label_dir, f"{label}_{int(time.time()*1000)}_{i}.png")
        cv2.imwrite(output_path, cv2.resize(roi, (28,28)))
        saved += 1
    return saved

def load_model():
    if not tf_available:
        return None
    if os.path.exists(MODEL_FILE):
        try:
            return load_model(MODEL_FILE)
        except Exception as e:
            print("model load error:", e)
            return None
    return None

def predict_digit(model, roi_array):
    img = cv2.resize(roi_array, (32,32))
    img = img.astype('float32')/255.0
    img = img.reshape(1,32,32,1)
    predictions = model.predict(img)
    digit = str(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))
    return digit, confidence

def check_login_result(driver):
    time.sleep(0.5)
    try:
        body_text = driver.find_element(By.TAG_NAME, "body").text.lower()
        
        if "wrong code" in body_text or "incorrect code" in body_text:
            print("  detected: wrong code")
            return False
            
        if "username and password don't match" in body_text or "invalid credentials" in body_text:
            print("  detected: wrong credentials but code was correct")
            return True
        
        if "dashboard" in body_text or "profile" in body_text or "logout" in body_text:
            print("  detected: successful login")
            return True
            
        print("  no specific message detected")
        return False
    except Exception as e:
        print(f"  error checking result: {e}")
        return False

def safe_form_input(driver, wait, username, password, captcha_text, wait_time=SUBMIT_WAIT):
    try:
        username_field = wait.until(EC.element_to_be_clickable((By.NAME, "loginUsername")))
        password_field = wait.until(EC.element_to_be_clickable((By.NAME, "loginPassword")))
        captcha_field = wait.until(EC.element_to_be_clickable((By.NAME, "norobot_login")))
    except Exception as e:
        print("safe_input: couldn't find form fields:", e)
        return False

    try:
        username_field.clear(); username_field.click()
        password_field.clear(); password_field.click()
    except Exception:
        pass

    username_field.send_keys(username)
    password_field.send_keys(password)
    time.sleep(0.08)

    input_success = False
    for attempt in range(MAX_TYPING_ATTEMPTS):
        try:
            try:
                captcha_field.click()
            except Exception:
                pass
            try:
                captcha_field.clear()
            except Exception:
                try:
                    driver.execute_script("arguments[0].value = '';", captcha_field)
                except Exception:
                    pass

            for ch in captcha_text:
                captcha_field.send_keys(ch)
                time.sleep(TYPING_DELAY)

            time.sleep(0.12)

            try:
                current_value = captcha_field.get_attribute('value') or ""
            except Exception:
                current_value = ""
            if current_value.strip() == captcha_text:
                input_success = True
                break
            else:
                try:
                    driver.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('input',{bubbles:true})); arguments[0].dispatchEvent(new Event('change',{bubbles:true}));", captcha_field, captcha_text)
                    time.sleep(0.08)
                    val2 = captcha_field.get_attribute('value') or ""
                    if val2.strip() == captcha_text:
                        input_success = True
                        break
                except Exception:
                    pass

                print(f"typing attempt {attempt+1} incomplete (got '{current_value}'), retrying...")
                time.sleep(0.12)
        except Exception as e:
            print("exception during captcha typing:", e)
            time.sleep(0.12)

    if not input_success:
        try:
            driver.execute_script("arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('input',{bubbles:true})); arguments[0].dispatchEvent(new Event('change',{bubbles:true}));", captcha_field, captcha_text)
            time.sleep(0.08)
            val3 = captcha_field.get_attribute('value') or ""
            if val3.strip() == captcha_text:
                input_success = True
        except Exception:
            pass

    try:
        try:
            submit_button = wait.until(EC.element_to_be_clickable((By.ID, "register")))
        except Exception:
            submit_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']")))
        time.sleep(0.08)
        submit_button.click()
        time.sleep(wait_time)
        return True
    except Exception as e:
        print("safe_input: submit error:", e)
        return False

def refresh_captcha_image(driver, captcha_element, url=URL, timeout=CAPTCHA_REFRESH_TIMEOUT):
    try:
        prev_src = captcha_element.get_attribute('src')
    except Exception:
        prev_src = None
    temp_before = os.path.join(DATA_FOLDER, "temp_before.png")
    try:
        captcha_element.screenshot(temp_before)
    except Exception:
        temp_before = None
    prev_hash = get_file_hash(temp_before) if temp_before else None

    refresh_selectors = [
        "//button[contains(@class,'refresh') or contains(@id,'refresh') or contains(@aria-label,'refresh')]",
        "//a[contains(@class,'refresh') or contains(@id,'refresh')]",
        "//button[contains(@class,'reload') or contains(@id,'reload')]",
        "//a[contains(@class,'reload')]",
        "//span[contains(@class,'reload') or contains(@id,'reload')]",
        "//i[contains(@class,'refresh') or contains(@class,'reload')]",
        "//button[contains(., 'refresh') or contains(., 'reload')]",
        "//a[contains(., 'refresh') or contains(., 'reload')]",
    ]
    attempted_actions = []
    for selector in refresh_selectors:
        try:
            buttons = driver.find_elements(By.XPATH, selector)
            for btn in buttons:
                try:
                    btn.click()
                    attempted_actions.append(("click", selector))
                    time.sleep(0.2)
                except Exception:
                    pass
        except Exception:
            pass

    try:
        captcha_element.click()
        attempted_actions.append(("click", "captcha_element"))
        time.sleep(0.2)
    except Exception:
        pass

    try:
        driver.refresh()
        attempted_actions.append(("driver.refresh", ""))
        time.sleep(3)
    except Exception:
        try:
            driver.get(url)
            attempted_actions.append(("driver.get", url))
            time.sleep(2)
        except Exception:
            pass

    start_time = time.time()
    new_src = prev_src
    new_hash = prev_hash
    changed = False
    while time.time() - start_time < timeout:
        try:
            current_element = driver.find_element(By.XPATH, '//img[@alt="captcha"]')
            new_src = current_element.get_attribute('src')
            temp_after = os.path.join(DATA_FOLDER, "temp_after.png")
            try:
                current_element.screenshot(temp_after)
                new_hash = get_file_hash(temp_after)
            except Exception:
                new_hash = None
            if prev_src and new_src and prev_src != new_src:
                changed = True
                captcha_element = current_element
                break
            if prev_hash and new_hash and prev_hash != new_hash:
                changed = True
                captcha_element = current_element
                break
        except Exception:
            pass
        time.sleep(0.3)

    try:
        if temp_before and os.path.exists(temp_before):
            os.remove(temp_before)
        temp_after = os.path.join(DATA_FOLDER, "temp_after.png")
        if os.path.exists(temp_after):
            os.remove(temp_after)
    except Exception:
        pass

    return captcha_element, changed, attempted_actions

chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
driver = webdriver.Chrome(options=chrome_options)
driver.get(URL)

try:
    wait = WebDriverWait(driver, 10)
    model = load_model()
    print("model loaded:", bool(model))
    session_count = 0

    while session_count < TOTAL_SESSIONS:
        session_count += 1
        print(f"\nsession {session_count}/{TOTAL_SESSIONS}")
        driver.get(URL)
        time.sleep(0.6)
        captcha_element = wait.until(EC.presence_of_element_located((By.XPATH, '//img[@alt="captcha"]')))
        timestamp = int(time.time()*1000)
        captcha_file = os.path.join(DATA_FOLDER, "captcha_images", f"captcha_{timestamp}.png")
        captcha_element.screenshot(captcha_file)
        print("saved captcha:", captcha_file)

        attempt_count = 0
        success = False
        last_prediction = ""
        method_used = "none"
        confidence_score = 0

        while attempt_count < MAX_RETRIES and not success:
            attempt_count += 1
            print(f" attempt {attempt_count}/{MAX_RETRIES}")

            regions = extract_digit_regions(captcha_file, debug_prefix=os.path.join(DATA_FOLDER, f"debug_{timestamp}_{attempt_count}"))
            predicted = ""
            confidences = []

            if model is not None and regions:
                for roi in regions:
                    digit, conf = predict_digit(model, roi)
                    if conf < 0.10:
                        text, text_conf = extract_text_with_tesseract(Image.fromarray(roi),
                                                               "--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789")
                        predicted += text if text else "?"
                        confidences.append(text_conf if text_conf>0 else 0)
                    else:
                        predicted += digit
                        confidences.append(int(conf*100))
                method_used = "model+fallback"
                confidence_score = sum(confidences)/len(confidences) if confidences else 0
            else:
                if regions:
                    for roi in regions:
                        text, text_conf = extract_text_with_tesseract(Image.fromarray(roi),
                                                               "--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789")
                        predicted += text if text else "?"
                        confidences.append(text_conf if text_conf>0 else 0)
                else:
                    raw_text, text_conf = extract_text_with_tesseract(Image.open(captcha_file),
                                                           "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789")
                    predicted = raw_text
                    confidences.append(text_conf)
                method_used = "tesseract"
                confidence_score = sum(confidences)/len(confidences) if confidences else 0

            predicted = ''.join(filter(str.isdigit, predicted))[:EXPECTED_LENGTH]
            
            if len(predicted) < EXPECTED_LENGTH:
                print(f"  only {len(predicted)} digits detected, trying full image...")
                raw_text, text_conf = extract_text_with_tesseract(Image.open(captcha_file),
                                                       "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789")
                if len(raw_text) >= len(predicted):
                    predicted = ''.join(filter(str.isdigit, raw_text))[:EXPECTED_LENGTH]
                    confidence_score = text_conf
                    method_used += "+fullimage"
            
            last_prediction = predicted
            print("  predicted ->", predicted, "method:", method_used, "confidence:", confidence_score)

            form_submitted = safe_form_input(driver, wait, USERNAME, PASSWORD, predicted, wait_time=SUBMIT_WAIT)
            if not form_submitted:
                print("  warning: form submission failed. logging and continuing.")
                with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), captcha_file, predicted, method_used, confidence_score, False, "submit_failed"])
                try:
                    current_captcha = driver.find_element(By.XPATH, '//img[@alt="captcha"]')
                except Exception:
                    current_captcha = None
                if current_captcha:
                    new_element, changed, actions = refresh_captcha_image(driver, current_captcha, url=URL, timeout=CAPTCHA_REFRESH_TIMEOUT)
                    if changed:
                        timestamp = int(time.time()*1000)
                        captcha_file = os.path.join(DATA_FOLDER, "captcha_images", f"captcha_{timestamp}.png")
                        try:
                            new_element.screenshot(captcha_file)
                        except Exception:
                            pass
                continue

            success = check_login_result(driver)
            if success:
                print("  success: captcha was correct.")
                with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), captcha_file, predicted, method_used, confidence_score, True, ""])
                if AUTO_SAVE_SUCCESS and regions and len(regions) == len(predicted):
                    saved_count = save_labeled_data(regions, predicted)
                    print(f"  saved {saved_count} labeled regions from successful attempt.")
                time.sleep(0.4)
                driver.get(URL)
                time.sleep(0.8)
                break
            else:
                print("  failed: captcha was incorrect.")
                with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), captcha_file, predicted, method_used, confidence_score, False, "captcha_incorrect"])
                unlabeled_path = os.path.join(DATA_FOLDER, "unlabeled", f"failed_{timestamp}_{attempt_count}.png")
                cv2.imwrite(unlabeled_path, cv2.imread(captcha_file))

                try:
                    current_captcha = driver.find_element(By.XPATH, '//img[@alt="captcha"]')
                except Exception:
                    current_captcha = None
                if current_captcha:
                    print("  attempting to refresh captcha...")
                    new_element, changed, actions = refresh_captcha_image(driver, current_captcha, url=URL, timeout=CAPTCHA_REFRESH_TIMEOUT)
                    print("   refresh attempted:", actions, "changed=", changed)
                    if changed:
                        timestamp = int(time.time()*1000)
                        captcha_file = os.path.join(DATA_FOLDER, "captcha_images", f"captcha_{timestamp}.png")
                        try:
                            new_element.screenshot(captcha_file)
                            print("   new captcha saved:", captcha_file)
                        except Exception:
                            driver.get(URL)
                            time.sleep(0.6)
                            current_element = driver.find_element(By.XPATH, '//img[@alt="captcha"]')
                            current_element.screenshot(captcha_file)
                            print("   new captcha saved after reload:", captcha_file)
                    else:
                        print("   captcha didn't change. forcing page reload.")
                        driver.get(URL)
                        time.sleep(0.8)
                        current_element = driver.find_element(By.XPATH, '//img[@alt="captcha"]')
                        captcha_file = os.path.join(DATA_FOLDER, "captcha_images", f"captcha_{int(time.time()*1000)}.png")
                        current_element.screenshot(captcha_file)
                        print("   new captcha (reload):", captcha_file)
                else:
                    print("  couldn't find captcha element for refresh; reloading page.")
                    driver.get(URL)
                    time.sleep(0.8)
                    current_element = driver.find_element(By.XPATH, '//img[@alt="captcha"]')
                    captcha_file = os.path.join(DATA_FOLDER, "captcha_images", f"captcha_{int(time.time()*1000)}.png")
                    current_element.screenshot(captcha_file)
                    print("   new captcha (reload):", captcha_file)

                time.sleep(0.35)

        time.sleep(0.3)

    print("completed all sessions.")
except Exception as e:
    print("error:", e)
finally:
    driver.quit()
