import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# ======================== 설정 ========================
CSV_PATH = "new_analysis_output/all_classes/60x60/all_classes__60x60_features.csv"
SELECTED_CLASSES = ["clean", "transparent", "semi_transparent", "opaque"]
MODEL_SAVE_PATH = "dirty_vs_clean_rf_model_60.pkl"
SCALER_SAVE_PATH = "dirty_vs_clean_scaler_60.pkl"

# ======================== 사용 feature 정의 ========================
top5_features = [
    "laplacian_var",
    "dwt_HL_energy",
    "dwt_LH_energy",
    "fft_highfreq_ratio",
    "dwt_HH_energy",
    "saturation_mean"
]

# ======================== 데이터 로드 ========================
df = pd.read_csv(CSV_PATH)
df = df[df['class'].isin(SELECTED_CLASSES)].copy()
df['binary_class'] = df['class'].apply(lambda x: 'dirty' if x != 'clean' else 'clean')

# 정규화
scaler = StandardScaler()
df[top5_features] = scaler.fit_transform(df[top5_features].fillna(0))
joblib.dump(scaler, SCALER_SAVE_PATH)

# 학습 데이터 준비
X = df[top5_features]
y = df['binary_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# 모델 학습
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# 평가
print("\n[RandomForest 성능 (Top Feature 사용)]\n")
print(classification_report(y_test, y_pred))
joblib.dump(rf, MODEL_SAVE_PATH)
print(f"\n[모델 저장 완료] => {MODEL_SAVE_PATH}")