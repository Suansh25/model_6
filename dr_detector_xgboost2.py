import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import frangi
from xgboost import XGBClassifier
#feature extraction feature using the following various features and methods
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    entropy = shannon_entropy(gray)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    lbp_hist = lbp_hist.tolist()

    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    vessels = frangi(gray)
    vessel_area = np.sum(vessels > 0.5) / vessels.size

    return [entropy, contrast, homogeneity, energy, correlation, blur, vessel_area] + lbp_hist

def load_data_from_folders(root_dir):
    features = []
    labels = []

    print("\n🔄 Extracting features from images...")
    for label_name in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            feats = extract_features(image)
            features.append(feats)
            labels.append(label_name)

    return np.array(features), np.array(labels)

def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=10,
        objective="multi:softmax",
        num_class=5,
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n✅ Model Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, "j48_xgb_model.pkl")
    joblib.dump(encoder, "label_encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\n💾 Model, encoder, and scaler saved as 'j48_xgb_model.pkl', 'label_encoder.pkl', and 'scaler.pkl'")

if __name__ == "__main__":
    dataset_dir = "dataset"
    X, y = load_data_from_folders(dataset_dir)
    train_model(X, y)
