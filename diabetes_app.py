import streamlit as st
import pandas as pd
import pickle

# 1. 의료 대시보드 스타일의 페이지 설정
st.set_page_config(
    page_title="의료 진단 보조 시스템 (CDSS)", 
    page_icon="🩺",
    layout="wide"
)

# 메디컬 톤 CSS 커스텀
st.markdown("""
    <style>
    .report-title { font-size: 2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 5px; }
    .report-subtitle { font-size: 1rem; color: #4B5563; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="report-title">🩺 당뇨병 발병 위험도 AI 진단 보조 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="report-subtitle">Clinical Decision Support System (CDSS) - 멀티 파일 로드 버젼</div>', unsafe_allow_html=True)
st.markdown("---")


# 2. 모델 및 스케일러 파일 각각 로드 (캐싱 적용)
@st.cache_resource
def load_medical_artifacts():
    try:
        # 모델 로드
        with open("Diabetes_model.pkl", "rb") as f:
            model = pickle.load(f)
            # 만약 기존 pkl 안에 딕셔너리로 묶여있던 경우를 위한 안전장치
            if isinstance(model, dict): 
                model = model.get('model', model)
            elif isinstance(model, (list, tuple)):
                model = model[0]
                
        # 💡 제공해주신 스케일러 파일 로드
        with open("diabetes_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
            if isinstance(scaler, dict):
                scaler = scaler.get('scaler', scaler)
            elif isinstance(scaler, (list, tuple)) and len(scaler) > 1:
                scaler = scaler[1]

        return model, scaler
    except FileNotFoundError as e:
        st.error(f"⚠️ 필수 파일을 찾을 수 없습니다: {e.filename}")
        return None, None

model, scaler = load_medical_artifacts()

if model is None or scaler is None:
    st.warning("🚨 `Diabetes_model.pkl`과 `diabetes_scaler.pkl` 파일이 현재 실행 스크립트와 같은 폴더에 있는지 확인해주세요.")
    st.stop()


# 3. 의학적 기준에 따른 입력 폼 분할
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📋 1. 기초 문진 및 신체 계측 (Anthro & History)")
    age = st.number_input("연령 (Age)", min_value=1, max_value=120, value=30, step=1)
    pregnancies = st.number_input("임신 횟수 (Pregnancies)", min_value=0, max_value=20, value=0, step=1)
    bmi = st.number_input("체질량지수 (BMI)", min_value=0.0, max_value=70.0, value=23.0, step=0.1, format="%.1f")
    skin_thickness = st.number_input("삼두근 피부 두께 (Triceps Skin Thickness, mm)", min_value=0.0, max_value=100.0, value=20.0, step=0.5)

with col_right:
    st.subheader("🧪 2. 임상 검사 데이터 (Laboratory Results)")
    glucose = st.number_input("공복 혈당 (Plasma Glucose, mg/dL)", min_value=0.0, max_value=400.0, value=100.0, step=1.0)
    blood_pressure = st.number_input("이완기 혈압 (Diastolic Blood Pressure, mmHg)", min_value=0.0, max_value=200.0, value=80.0, step=1.0)
    insulin = st.number_input("혈청 인슐린 (2-Hour Serum Insulin, mu U/ml)", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
    dpf = st.number_input("당뇨 가족력 가중치 (Diabetes Pedigree Function)", min_value=0.0, max_value=3.0, value=0.5, step=0.001, format="%.3f")

st.markdown("---")


# 4. 진단 결과 분석 및 시각화
if st.button("🩺 종합 검사 결과 분석 (Run Analysis)", use_container_width=True, type="primary"):
    
    # 원본 데이터프레임 빌드 (학습 때 사용한 순서 그대로 매칭)
    input_data = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
        columns=['임신횟수', '혈당', '혈압', '피부두께', '인슐린', 'BMI', '당뇨내력가중치', '나이']
    )
    
    # 💡 로드된 외부 스케일러로 안전하게 변환 진행
    input_data_scaled = scaler.transform(input_data)

    # 예측 및 확률 계산
    predicted = model.predict(input_data_scaled)[0]
    prob = model.predict_proba(input_data_scaled)[0][1] * 100
    
    # 결과 섹션 레이아웃
    st.markdown("### 📊 AI 기반 임상 진단 분석 리포트")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(label="입력된 혈당 수치", value=f"{glucose} mg/dL", delta="주의" if glucose >= 126 else "정상", delta_color="inverse")
        st.metric(label="환자 BMI 지수", value=f"{bmi}", delta="비만" if bmi >= 25 else "정상", delta_color="inverse")

    with res_col2:
        if predicted == 1:
            st.error(f"🚨 **[고위험군 소견] 당뇨병 발병 가능성이 높습니다.**")
            st.markdown(f"""
            * **분석된 위험 확률:** `{prob:.1f}%`
            * **임상적 권고:** 본 결과는 불러온 정적 스케일러 지표를 거쳐 계산된 검증 데이터입니다. 환자의 수치가 임상적 위험선 상에 있으므로 정밀 검사 유도가 권장됩니다.
            """)
            st.progress(int(prob))
        else:
            st.success(f"✅ **[정상 소견] 당뇨병 발병 위험도가 낮습니다.**")
            st.markdown(f"""
            * **분석된 위험 확률:** `{prob:.1f}%`
            * **임상적 권고:** 본 환자는 현재 데이터 기준 안정권군으로 분류됩니다. 정기적인 추적 관찰을 권장합니다.
            """)
            st.progress(int(prob))