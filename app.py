# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import joblib
import io
import base64

from flask import Flask, request, jsonify, render_template
import matplotlib.pyplot as plt
import seaborn as sns

# --- [초기 설정] 서버가 시작될 때 한번만 실행됩니다 ---
print("--- [시작] 예측 서버(Scikit-learn) 초기 설정을 시작합니다. ---")

app = Flask(__name__)

# 1단계에서 저장한 결과물들 로드
try:
    # .h5 대신 .pkl 모델 파일을 불러옵니다.
    model = joblib.load('player_aging_curve_model.pkl')
    scaler = joblib.load('scaler.pkl')
    combined_df = pd.read_csv('combined_fifa_data_processed.csv', low_memory=False)
    
    with open('final_features.json', 'r') as f:
        final_features = json.load(f)
    with open('numerical_cols_to_scale.json', 'r') as f:
        numerical_cols_to_scale = json.load(f)
        
    print("--- ✅ 모델 및 설정 파일 로드 성공! ---")
except Exception as e:
    print(f"--- ❌ 파일 로드 실패: {e} ---")
    print("--- ❌ 서버를 시작하기 전에 train.py를 먼저 실행해야 합니다. ---")


# 예측을 수행하는 핵심 함수 (ID 불일치 문제 해결된 최종 버전)
def predict_player_form_change(player_name, max_target_year=2030):
    # 1. 입력된 이름으로 모든 고유 player_id를 찾습니다.
    matching_ids = combined_df[combined_df['player_name'] == player_name]['player_id'].unique()

    if len(matching_ids) == 0:
        return None, []

    # 2. 찾아낸 모든 player_id에 해당하는 모든 데이터를 불러옵니다.
    player_history = combined_df[combined_df['player_id'].isin(matching_ids)].sort_values(by='year').copy()
    player_history.drop_duplicates(subset=['year'], keep='last', inplace=True)

    if player_history.empty:
        return None, []

    # 선수의 고점(Peak) 정보 찾기
    peak_row = player_history.loc[player_history['overall_rating'].idxmax()]
    peak_age = int(peak_row['age'])
    peak_year = int(peak_row['year'])
    
    # 시뮬레이션 데이터 기록용 리스트
    all_simulated_data = []

    # 과거 데이터 기록
    player_history['actual_annual_performance_change'] = player_history['overall_rating'].diff()
    player_history['actual_cumulative_performance_change'] = player_history['actual_annual_performance_change'].fillna(0).cumsum()

    for _, row in player_history.iterrows():
        all_simulated_data.append({
            'year': int(row['year']),
            'age': int(row['age']),
            'performance_change_predicted': row['actual_annual_performance_change'],
            'cumulative_performance_change': row['actual_cumulative_performance_change'],
            'is_actual': True
        })

    # 예측 시작점 설정
    latest_data = player_history.iloc[-1].copy()
    current_year = int(latest_data['year'])
    current_age = int(latest_data['age'])
    cumulative_performance_change = latest_data['actual_cumulative_performance_change'] if pd.notna(latest_data['actual_cumulative_performance_change']) else 0

    # 모델이 학습한 피처(final_features)와 동일한 구조의 DataFrame을 생성
    predict_df = pd.DataFrame(columns=final_features)
    predict_df.loc[0] = 0

    # 숫자형 데이터 채우기
    for col in numerical_cols_to_scale:
        if col in latest_data and pd.notna(latest_data[col]):
            predict_df.loc[0, col] = latest_data[col]
            
    # 범주형 데이터 원-핫 인코딩 처리
    categorical_original_features = ['nationality', 'club_name', 'preferred_foot', 'work_rate', 'body_type']
    for col in categorical_original_features:
        if col in latest_data and pd.notna(latest_data[col]):
            value = latest_data[col]
            one_hot_col = f"{col}_{value}"
            if one_hot_col in predict_df.columns:
                predict_df.loc[0, one_hot_col] = 1
    
    # 미래 예측 시뮬레이션
    for i in range(1, max_target_year - current_year + 1):
        simulated_year = current_year + i
        simulated_age = current_age + i

        predict_df['age'] = float(simulated_age)
        predict_df['year'] = float(simulated_year)

        predict_df_scaled = predict_df.copy()
        predict_df_scaled[numerical_cols_to_scale] = scaler.transform(predict_df[numerical_cols_to_scale])
        
        # Scikit-learn 모델의 predict 사용
        predicted_change = model.predict(predict_df_scaled[final_features])[0]
        cumulative_performance_change += predicted_change

        all_simulated_data.append({
            'year': simulated_year,
            'age': simulated_age,
            'performance_change_predicted': predicted_change,
            'cumulative_performance_change': cumulative_performance_change,
            'is_actual': False
        })
        
        predict_df.loc[0, 'overall_rating'] += predicted_change

    # 에이징 커브 시작 시점 찾기
    aging_curve_start_age = None
    aging_curve_start_year = None
    for data_point in all_simulated_data:
        if not data_point['is_actual'] and data_point['performance_change_predicted'] < 0:
            aging_curve_start_age = data_point['age']
            aging_curve_start_year = data_point['year']
            break

    # 예측 요약 정보 생성
    prediction_summary = {
        'player_name': player_name,
        'current_age': current_age,
        'peak_age': peak_age,
        'peak_year': peak_year,
        'aging_curve_start_age': aging_curve_start_age,
        'aging_curve_start_year': aging_curve_start_year
    }
    
    return prediction_summary, all_simulated_data


# --- [API 라우팅] 사용자의 요청을 처리하는 부분 ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    player_name = data.get('player_name')

    if not player_name:
        return jsonify({'error': '선수 이름이 없습니다.'}), 400

    summary, sim_data = predict_player_form_change(player_name)

    if summary is None:
        return jsonify({'error': f"'{player_name}' 선수의 데이터를 찾을 수 없습니다."}), 404
        
    # 그래프 생성
    sim_df = pd.DataFrame(sim_data)
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0d1f0c')
    ax.set_facecolor('#133311')
    
    actual_data = sim_df[sim_df['is_actual'] == True]
    predicted_data = sim_df[sim_df['is_actual'] == False]
    
    if not actual_data.empty and not predicted_data.empty:
        last_actual = actual_data.iloc[-1:]
        bridge_data = pd.concat([last_actual, predicted_data.iloc[:1]])
        
        sns.lineplot(ax=ax, data=actual_data, x='year', y='cumulative_performance_change', marker='o', color='#33bbff', label='Actual', linewidth=2.5)
        sns.lineplot(ax=ax, data=bridge_data, x='year', y='cumulative_performance_change', color='#ffcc00', linestyle='--')
        sns.lineplot(ax=ax, data=predicted_data, x='year', y='cumulative_performance_change', marker='X', markersize=8, color='#ffcc00', linestyle='--', label='Predicted', linewidth=2.5)
    else:
         sns.lineplot(ax=ax, data=sim_df, x='year', y='cumulative_performance_change', hue='is_actual', style='is_actual', markers=True, dashes=False)

    ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.set_xlabel("Year", color='white', fontsize=12)
    ax.set_ylabel("Cumulative Performance Change", color='white', fontsize=12)
    ax.set_title(f"{player_name} Performance Curve", color='white', fontsize=16, fontweight='bold')
    
    legend = ax.legend()
    legend.get_frame().set_facecolor('#133311')
    legend.get_frame().set_edgecolor('white')
    for text in legend.get_texts():
        text.set_color('white')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)

    return jsonify({
        'summary': summary,
        'plot_image': f'data:image/png;base64,{plot_url}'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)