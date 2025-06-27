import os
import cv2
import numpy as np
import pandas as pd
import joblib
import pywt

# ======================== 설정 ========================
TEST_IMAGE_DIR = "data/pig/"
SAVE_DIR = "final_test/dirty_vs_clean_60_pig"
MODEL_PATH = "dirty_vs_clean_rf_model_60.pkl"
SCALER_PATH = "dirty_vs_clean_scaler_60.pkl"
OVERLAY_COLOR = (0, 0, 255)
ALPHA = 0.4
BLOCK_DIV_X = 40
BLOCK_DIV_Y = 40
PADDING_BLOCKS = 1
PADDING_BOTTOM_BLOCKS = 1

# ======================== 사용 feature 정의 ========================
top5_features = [
    "laplacian_var",
    "dwt_HL_energy",
    "dwt_LH_energy",
    "fft_highfreq_ratio",
    "dwt_HH_energy",
    "saturation_mean"
]

os.makedirs(SAVE_DIR, exist_ok=True)
rf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ======================== feature 추출 함수 ========================
def extract_features(block):
    gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
    coeffs2 = pywt.dwt2(gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    fft = np.fft.fft2(gray)
    fft_high = np.sum(np.abs(fft) > 20) / gray.size
    feats = {
        "laplacian_var": np.var(cv2.Laplacian(gray, cv2.CV_64F)),
        "dwt_HL_energy": np.mean(np.square(HL)),
        "dwt_LH_energy": np.mean(np.square(LH)),
        "fft_highfreq_ratio": fft_high,
        "dwt_HH_energy": np.mean(np.square(HH)),
        "saturation_mean": np.mean(hsv[:, :, 1])
    }
    return pd.DataFrame([feats])[top5_features]

# ======================== 테스트 이미지 처리 ========================
for fname in os.listdir(TEST_IMAGE_DIR):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    rgb_path = os.path.join(TEST_IMAGE_DIR, fname)
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        continue

    IMG_HEIGHT, IMG_WIDTH = rgb.shape[:2]
    BLOCK_WIDTH = IMG_WIDTH // BLOCK_DIV_X
    BLOCK_HEIGHT = IMG_HEIGHT // BLOCK_DIV_Y
    CROP_LEFT = PADDING_BLOCKS * BLOCK_WIDTH
    CROP_RIGHT = IMG_WIDTH - PADDING_BLOCKS * BLOCK_WIDTH
    CROP_TOP = PADDING_BLOCKS * BLOCK_HEIGHT
    CROP_BOTTOM = IMG_HEIGHT - PADDING_BOTTOM_BLOCKS * BLOCK_HEIGHT

    overlay = rgb.copy()

    for j in range(BLOCK_DIV_Y):
        for i in range(BLOCK_DIV_X):
            x = i * BLOCK_WIDTH
            y = j * BLOCK_HEIGHT

            if (x < CROP_LEFT or x + BLOCK_WIDTH > CROP_RIGHT or
                y < CROP_TOP or y + BLOCK_HEIGHT > CROP_BOTTOM):
                continue

            block = rgb[y:y+BLOCK_HEIGHT, x:x+BLOCK_WIDTH]
            if block.size == 0:
                continue

            feature_df = extract_features(block).fillna(0)
            feature_scaled = scaler.transform(feature_df)
            pred = rf.predict(pd.DataFrame(feature_scaled, columns=top5_features))[0]

            if pred == 'dirty':
                cv2.rectangle(overlay, (x, y), (x + BLOCK_WIDTH, y + BLOCK_HEIGHT), OVERLAY_COLOR, -1)

    blended = cv2.addWeighted(overlay, ALPHA, rgb, 1 - ALPHA, 0)
    cropped = blended[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT]
    save_path = os.path.join(SAVE_DIR, fname)
    cv2.imwrite(save_path, cropped)
    print(f"Saved: {save_path}")
