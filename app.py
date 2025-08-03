import pandas as pd
import numpy as np
import json
import joblib # tensorflow 대신 joblib 사용
import io
import base64
from flask import Flask, request, jsonify, render_template
# import tensorflow as tf # 더 이상 필요 없음
import matplotlib.pyplot as plt
import seaborn as sns

print("--- [시작] 예측 서버(Scikit-learn) 초기 설정을 시작합니다. ---")
app = Flask(__name__)

try:
    # .h5 대신 .pkl 모델 파일을 불러옵니다.
    model = joblib.load('player_aging_curve_model.pkl')
    scaler = joblib.load('scaler.pkl')
    # ... (나머지 파일 로드는 이전과 동일) ...
    print("--- ✅ 모델 및 설정 파일 로드 성공! ---")
except Exception as e:
    print(f"--- ❌ 파일 로드 실패: {e} ---")

# --- (predict_player_form_change 함수와 API 라우팅 부분은 이전과 동일) ---
# (TensorFlow를 사용하지 않으므로, 이전 버전의 코드를 그대로 사용해도 됩니다.)

def predict_api():
    # ... (이전 코드와 동일) ...
    # 예측 부분만 살짝 바뀔 수 있습니다.
    predicted_change = model.predict(predict_df_scaled[final_features])[0] # Keras와 달리 [0][0]이 아님
    # ... (나머지는 동일) ...