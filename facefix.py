import os
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from skimage.feature import local_binary_pattern
from skimage.color import rgb2hsv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# === CONSTANTS ===
LBP_RADIUS = 1
LBP_N_POINTS = 8
DATASET_DIR = "dermnet_dataset"
CATEGORIES = ['acne', 'eczema', 'psoriasis']  # Use any categories available on dermnetnz


# === FEATURE EXTRACTION FUNCTION ===
def extract_features(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)

    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    hist_lbp = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_N_POINTS + 3), density=True)[0]

    hue = hsv[:, :, 0].mean() / 180.0
    saturation = hsv[:, :, 1].mean() / 255.0
    value = hsv[:, :, 2].mean() / 255.0

    return np.concatenate([hist_lbp, [hue, saturation, value]])


# === STEP 3: DOWNLOAD IMAGES FROM DERmNET ===
def download_images_from_dermnet(category, limit=50):
    url = f"https://dermnetnz.org/topics/{category}"
    save_dir = os.path.join(DATASET_DIR, category)
    os.makedirs(save_dir, exist_ok=True)

    print("Files downloaded:")
    for cat in CATEGORIES:
        print(cat, "->", os.listdir(os.path.join(DATASET_DIR, cat)))

    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.content, 'html.parser')
    imgs = soup.find_all("img")

    count = 0
    for img in imgs:
        src = img.get("src")
        if src and src.startswith("/static/"):
            img_url = f"https://dermnetnz.org{src}"
            try:
                img_data = requests.get(img_url).content
                with open(f"{save_dir}/{category}_{count}.jpg", "wb") as f:
                    f.write(img_data)
                count += 1
                if count >= limit:
                    break
            except:
                pass


# === DOWNLOAD DATASET ===
for cat in CATEGORIES:
    download_images_from_dermnet(cat)

# === STEP 4: LOAD DATASET & EXTRACT FEATURES ===
data = []
labels = []

for label in CATEGORIES:
    folder = os.path.join(DATASET_DIR, label)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (100, 100))
            feat = extract_features(img)
            data.append(feat)
            labels.append(label)

X = np.array(data)
y = np.array(labels)

# === STEP 5: TRAIN MODEL ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel='rbf', probability=True)
model.fit(X_train_scaled, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"✅ Model training complete with accuracy: {accuracy:.4f}")

# === SAVE MODEL ===
joblib.dump(model, "skin_issue_model.joblib")
joblib.dump(scaler, "feature_scaler.joblib")
