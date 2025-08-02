# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import shutil
import joblib # 모델 객체 저장을 위해 추가
import json   # 피처 리스트 저장을 위해 추가

# 딥러닝 및 머신러닝 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

print("--- [시작] 1단계: 모델 학습 및 결과물 저장 프로세스를 시작합니다. ---")

# --- 데이터 전처리를 위한 헬퍼 함수 ---
def clean_value_wage(val):
    val = str(val).replace('€', '').replace(',', '')
    if 'M' in val:
        return float(val.replace('M', '')) * 1000000
    elif 'K' in val:
        return float(val.replace('K', '')) * 1000
    try:
        return float(val)
    except ValueError:
        return np.nan

def clean_height(height_str):
    if pd.isna(height_str): return np.nan
    s = str(height_str).lower().strip()
    if 'cm' in s: return float(s.replace('cm', ''))
    elif "'" in s and '"' in s:
        try:
            feet, inches = s.split("'")
            inches = inches.replace('"', '')
            total_inches = int(feet) * 12 + int(inches)
            return round(total_inches * 2.54, 2)
        except ValueError: return np.nan
    return np.nan

def clean_weight(weight_str):
    if pd.isna(weight_str): return np.nan
    s = str(weight_str).lower().strip()
    if 'kg' in s: return float(s.replace('kg', ''))
    elif 'lbs' in s: return round(float(s.replace('lbs', '')) * 0.453592, 2)
    try: return float(s)
    except ValueError: return np.nan


# --- Step 1: 데이터 로드 및 통합 ---
print("\n[1-1] FIFA 데이터셋 로딩 및 통합을 시작합니다...")

# 데이터 파일들이 저장된 경로 (./data/ 폴더)
data_path = './data/'
unzip_base_path = './unzipped_fifa_data/'

# 이전 실행으로 인한 잔여 폴더 삭제 및 재성성
if os.path.exists(unzip_base_path):
    shutil.rmtree(unzip_base_path)
os.makedirs(unzip_base_path, exist_ok=True)

# data 폴더에서 zip 파일 목록 찾기
zip_files = [f for f in os.listdir(data_path) if f.endswith('.zip')]

if not zip_files:
    print(f"경고: '{data_path}' 폴더에서 ZIP 파일을 찾을 수 없습니다. FIFA 데이터셋을 넣어주세요.")
    exit()

# ZIP 파일 압축 해제
for zip_file_name in zip_files:
    zip_file_path = os.path.join(data_path, zip_file_name)
    extract_to_folder = os.path.join(unzip_base_path, os.path.splitext(zip_file_name)[0].replace('.csv', ''))
    os.makedirs(extract_to_folder, exist_ok=True)
    try:
        shutil.unpack_archive(zip_file_path, extract_to_folder)
        print(f"✅ '{zip_file_name}' 압축 해제 성공")
    except Exception as e:
        print(f"❌ '{zip_file_name}' 압축 해제 실패: {e}")

# 압축 해제된 CSV 파일들 찾아서 합치기
all_dfs = []
for root, dirs, files in os.walk(unzip_base_path):
    for file_name in files:
        if file_name.endswith('.csv'):
            file_path = os.path.join(root, file_name)
            try:
                temp_df = pd.read_csv(file_path, low_memory=False)
                # 파일 이름에서 연도 정보 추출 (예: 'FIFA22' -> 2022)
                year_str = ''.join(filter(str.isdigit, file_name))
                if len(year_str) == 2: # 15, 16, ...
                    year = int("20" + year_str)
                elif len(year_str) == 4: # 2022, 2023, ...
                    year = int(year_str)
                else: # 연도 정보가 없는 경우를 대비
                    year = np.nan
                
                # 원본 데이터에 'year' 컬럼 추가
                if 'year' not in temp_df.columns:
                     # FIFA 15~23 데이터셋 이름 규칙을 활용
                    if '15' in file_name: temp_df['year'] = 2015
                    elif '16' in file_name: temp_df['year'] = 2016
                    elif '17' in file_name: temp_df['year'] = 2017
                    elif '18' in file_name: temp_df['year'] = 2018
                    elif '19' in file_name: temp_df['year'] = 2019
                    elif '20' in file_name: temp_df['year'] = 2020
                    elif '21' in file_name: temp_df['year'] = 2021
                    elif '22' in file_name: temp_df['year'] = 2022
                    elif '23' in file_name: temp_df['year'] = 2023

                all_dfs.append(temp_df)
                print(f"✅ '{file_name}' 로드 성공.")
            except Exception as e:
                print(f"❌ '{file_name}' 로드 실패: {e}")

# 모든 데이터프레임을 하나로 합치기
raw_combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"\n✅ 모든 데이터 통합 완료! 총 데이터 크기: {raw_combined_df.shape}")


# --- Step 2: 데이터 정제 및 컬럼 표준화 ---
print("\n[1-2] 데이터 정제 및 컬럼 표준화를 시작합니다...")
# 컬럼명을 소문자 및 snake_case로 변경
raw_combined_df.columns = raw_combined_df.columns.str.lower().str.replace(' ', '_')

# 컬럼명 매핑 (상이한 컬럼명을 하나로 통일)
CORE_COLS_MAP = {
    'player_id': ['sofifa_id', 'id'],
    'player_name': ['short_name', 'name', 'long_name'],
    'age': ['age'],
    'overall_rating': ['overall'],
    'potential': ['potential'],
    'market_value_eur': ['value_eur', 'value'],
    'wage_eur': ['wage_eur', 'wage'],
    'nationality': ['nationality_name', 'nationality'],
    'club_name': ['club_name', 'club'],
    'height_cm': ['height_cm', 'height'],
    'weight_kg': ['weight_kg', 'weight'],
    # ... 필요한 다른 컬럼들도 추가 ...
}

combined_df = pd.DataFrame()

for standard_name, possible_names in CORE_COLS_MAP.items():
    for col_name in possible_names:
        if col_name in raw_combined_df.columns:
            combined_df[standard_name] = raw_combined_df[col_name]
            break

# 원본 데이터에서 나머지 숫자형 특성들도 최대한 가져오기
for col in raw_combined_df.columns:
    if col not in combined_df.columns and pd.api.types.is_numeric_dtype(raw_combined_df[col]):
        combined_df[col] = raw_combined_df[col]

# 연도(year) 컬럼 복사
if 'year' in raw_combined_df.columns:
    combined_df['year'] = raw_combined_df['year']

# 데이터 클리닝 함수 적용
combined_df['market_value_eur'] = combined_df['market_value_eur'].apply(clean_value_wage)
combined_df['wage_eur'] = combined_df['wage_eur'].apply(clean_value_wage)
combined_df['height_cm'] = combined_df['height_cm'].apply(clean_height)
combined_df['weight_kg'] = combined_df['weight_kg'].apply(clean_weight)

print("✅ 데이터 정제 및 표준화 완료.")


# --- Step 3: 피처 엔지니어링 및 타겟 변수 생성 ---
print("\n[1-3] 피처 엔지니어링 및 타겟 변수 생성을 시작합니다...")
combined_df.sort_values(by=['player_id', 'year'], inplace=True)

# 예측에 사용할 숫자 특성 목록
PERFORMANCE_ATTRIBUTES = [
    'overall_rating', 'potential', 'crossing', 'finishing', 'heading_accuracy',
    'short_passing', 'volleys', 'dribbling', 'curve', 'fk_accuracy', 'long_passing',
    'ball_control', 'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance',
    'shot_power', 'jumping', 'stamina', 'strength', 'long_shots', 'aggression',
    'interceptions', 'positioning', 'vision', 'penalties', 'composure', 'marking',
    'standing_tackle', 'sliding_tackle'
]
# 실제 데이터에 있는 특성만 필터링
PERFORMANCE_ATTRIBUTES_FINAL = [attr for attr in PERFORMANCE_ATTRIBUTES if attr in combined_df.columns]

# 각 특성의 다음 해 변화량 계산
for attr in PERFORMANCE_ATTRIBUTES_FINAL:
    combined_df[f'next_year_{attr}_change'] = combined_df.groupby('player_id')[attr].diff().shift(-1)

change_cols = [f'next_year_{attr}_change' for attr in PERFORMANCE_ATTRIBUTES_FINAL]
# 타겟 변수 생성: 모든 변화량의 합
combined_df['next_year_performance_change'] = combined_df[change_cols].sum(axis=1)

# 다음 해 데이터 존재 여부 플래그
combined_df['has_next_year_data'] = combined_df['next_year_performance_change'].notna()
print("✅ 타겟 변수 'next_year_performance_change' 생성 완료.")


# --- Step 4: 모델 학습을 위한 데이터 준비 ---
print("\n[1-4] 모델 학습을 위한 데이터 준비를 시작합니다...")
# 다음 해 데이터가 있는 행만 필터링
model_data = combined_df[combined_df['has_next_year_data']].copy()
model_data.dropna(subset=['player_id', 'year'], inplace=True) # 필수 컬럼 결측치 제거

# 숫자형, 범주형 피처 식별 (예시, 실제 데이터에 맞게 조정 필요)
numerical_features = model_data.select_dtypes(include=np.number).columns.tolist()
categorical_features = ['nationality', 'club_name', 'preferred_foot', 'work_rate', 'body_type']
categorical_features = [f for f in categorical_features if f in model_data.columns]


# 제외할 컬럼 목록
excluded_from_features = [
    'player_id', 'player_name', 'has_next_year_data', 'next_year_performance_change'
] + change_cols # 변화량 컬럼들도 피처에서 제외

numerical_features = [f for f in numerical_features if f not in excluded_from_features]

# 결측치 처리
for col in numerical_features:
    model_data[col].fillna(model_data[col].median(), inplace=True)
for col in categorical_features:
    model_data[col].fillna('Unknown', inplace=True)

# 범주형 피처 원-핫 인코딩
model_data = pd.get_dummies(model_data, columns=categorical_features, dummy_na=False)

# 최종 피처(X)와 타겟(Y) 정의
final_features = [col for col in model_data.columns if col not in excluded_from_features and not col.startswith('next_year_')]
X = model_data[final_features]
Y = model_data['next_year_performance_change']

# 숫자형 피처 스케일링
numerical_cols_to_scale = [f for f in numerical_features if f in X.columns]
scaler = StandardScaler()
X[numerical_cols_to_scale] = scaler.fit_transform(X[numerical_cols_to_scale])

# 학습/테스트 데이터 분리
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"✅ 데이터 준비 완료. 학습 데이터: {X_train.shape[0]}개, 테스트 데이터: {X_test.shape[0]}개")
print(f"사용된 피처 개수: {X_train.shape[1]}")


# --- Step 5: 딥러닝 모델 구축 및 학습 ---
print("\n[1-5] 딥러닝 모델 구축 및 학습을 시작합니다...")
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()
history = model.fit(X_train, Y_train, epochs=30, batch_size=64, validation_split=0.2, verbose=1)

print("✅ 모델 학습 완료.")
loss, mae = model.evaluate(X_test, Y_test)
print(f"\n모델 평가 결과 -> Loss: {loss:.4f}, MAE: {mae:.4f}")


# --- Step 6: 결과물 저장 ---
print("\n[1-6] 학습된 모델과 주요 결과물을 파일로 저장합니다...")

# 1. 학습된 딥러닝 모델 저장
model.save('player_aging_curve_model.h5')
print("  - ✅ 모델 저장 완료: player_aging_curve_model.h5")

# 2. 데이터 스케일러 저장
joblib.dump(scaler, 'scaler.pkl')
print("  - ✅ 스케일러 저장 완료: scaler.pkl")

# 3. 모델에 사용된 피처(컬럼) 목록 저장
with open('final_features.json', 'w') as f:
    json.dump(final_features, f)
print("  - ✅ 피처 목록 저장 완료: final_features.json")
    
# 4. 스케일링에 사용된 숫자형 피처 목록 저장
with open('numerical_cols_to_scale.json', 'w') as f:
    json.dump(numerical_cols_to_scale, f)
print("  - ✅ 숫자형 피처 목록 저장 완료: numerical_cols_to_scale.json")
    
# 5. 예측 함수에 필요한 전체 데이터 저장
# (웹서버에서 특정 선수 데이터를 찾기 위해 필요)
combined_df.to_csv('combined_fifa_data_processed.csv', index=False)
print("  - ✅ 처리된 전체 데이터 저장 완료: combined_fifa_data_processed.csv")

shutil.rmtree(unzip_base_path) # 임시 압축해제 폴더 삭제
print(f"\n--- [완료] 1단계 프로세스가 성공적으로 끝났습니다. ---")