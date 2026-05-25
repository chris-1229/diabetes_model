import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. 페이지 설정
st.set_page_config(page_title="당뇨병 예측 프로그램", layout="centered")

# 2. 모델 및 스케일러 불러오기
# @st.cache_resource를 사용하면 앱이 실행될 때 한 번만 파일을 읽어오므로 속도가 빨라집니다.
@st.cache_resource
def load_models():
    try:
        model = joblib.load('Diabetes_model.pkl')
        scaler = joblib.load('diabetes_scalers.pkl')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"필수 파일을 찾을 수 없습니다: {e.filename}. 파일이 같은 디렉토리에 있는지 확인하세요.")
        return None, None

knn_model, scaler_8feat = load_models()

# 모델 로딩에 성공했을 때만 앱 실행
if knn_model is not None and scaler_8feat is not None:
    
    st.title("🩺 당뇨병 위험도 예측 시스템")
    st.write("아래 정보를 입력하시면 AI 모델이 당뇨병 가능성을 예측합니다.")
    st.markdown("---")

    # 3. 사용자 입력 받기 (레이아웃을 위해 2열로 구성)
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("임신 횟수", min_value=0, max_value=20, value=0, step=1)
        glucose = st.number_input("혈당 (Glucose)", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
        blood_pressure = st.number_input("혈압 (Blood Pressure)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        skin_thickness = st.number_input("피부 두께 (Skin Thickness)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        
    with col2:
        insulin = st.number_input("인슐린 (Insulin)", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
        bmi = st.number_input("BMI (체질량지수)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("당뇨 내력 가중치 (DPF)", min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.3f")
        age = st.number_input("나이", min_value=1, max_value=120, value=30, step=1)

    st.markdown("---")

    # 4. 예측 버튼 및 결과 출력
    if st.button("📊 당뇨병 여부 예측하기", type="primary"):
        
        # 입력 데이터를 DataFrame으로 변환 (기존 코드의 컬럼명과 순서 유지)
        input_data = pd.DataFrame(
            [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
            columns=['임신횟수', '혈당', '혈압', '피부두께', '인슐린', 'BMI', '당뇨내력가중치', '나이']
        )
        
        # 불러온 스케일러로 데이터 변환
        input_data_scaled = scaler_8feat.transform(input_data)
        
        # 예측 및 확률 계산
        predicted = knn_model.predict(input_data_scaled)
        prob = knn_model.predict_proba(input_data_scaled)
        
        diabetes_prob = prob[0][1] * 100  # 당뇨(1)일 확률
        
        # 결과 시각화 출력
        st.subheader("🔮 예측 결과")
        
        if predicted[0] == 1:
            st.error(f"⚠️ **당뇨병 위험군으로 예측됩니다.** (위험도: {diabetes_prob:.1f}%)")
        else:
            st.success(f"✅ **정상 범주로 예측됩니다.** (당뇨 가능성: {diabetes_prob:.1f}%)")
            
        # 게이지 바 형태로 확률 표시
        st.progress(int(diabetes_prob))