# ==========================================
# machine.py  (UPDATED - defect model -> joblib)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import pickle
import joblib  # ✅ 추가: joblib 로딩
import streamlit.components.v1 as components


# ==========================================
# 0. 수치 기반 모델용 피처 설정 (웨이퍼위치 제거, 19개)
# ==========================================
FEATURES = [
    '가로길이', '세로길이', '검출면적', '직경크기', '신호강도', '신호극성',
    '에너지값', '기준편차', '명도수준', '잡음정도', '중심거리', '방향각도',
    '정렬정도', '점형지수', '영역잡음', '상대강도', '활성지수', '패치신호', 'Aspect_Ratio'
]

MODEL_REAL_FAKE_PATH = r"lgbm_v4.pkl"
MODEL_DEFECT_PATH = r"best_defect_model.joblib"  # ✅ 변경: pkl -> joblib

LOG_FEATURES = [
    '가로길이', '세로길이', '검출면적', '직경크기', '신호강도',
    '에너지값', '기준편차', '명도수준', '잡음정도', '중심거리',
    '방향각도', '정렬정도', '점형지수', '영역잡음', '상대강도',
    '활성지수', '패치신호', 'Aspect_Ratio'
]

# 결함 라벨 매핑 (모델은 0~10 index를 내고, 실제 결함코드로 변환)
DEFECT_CLASS_LIST = [9, 10, 14, 17, 20, 21, 22, 28, 39, 56, 99]

def map_defect_index(idx: int) -> int:
    """모델의 index 예측값을 실제 결함코드로 매핑"""
    try:
        idx = int(idx)
    except:
        return idx
    if 0 <= idx < len(DEFECT_CLASS_LIST):
        return int(DEFECT_CLASS_LIST[idx])
    return idx


# ==========================================
# 1. YOLO 형상 분류용 클래스
# ==========================================
CLASS_NAMES = {
    0: 'Center', 1: 'Donut', 2: 'Edge-Loc', 3: 'Edge-Ring',
    4: 'Loc', 5: 'Near-full', 6: 'Random', 7: 'Scratch'
}

DEFECT_KNOWLEDGE_BASE = {
    'Center': {'korean': '센터 불량', 'cause': 'CBCMP', 'action': '이제/센터 구간 CMP 편차 여부 확인'},
    'Donut': {'korean': '도넛형 불량', 'cause': 'CBCMP', 'action': '패드 상태, 압력 조건, 슬러리 공급 균일성 점검'},
    'Edge-Loc': {'korean': '엣지 국부 불량', 'cause': 'PC, RMG', 'action': 'PC 공정 전·후 표면 클리닝 상태 점검, 설비 상태(온도, 압력, 회전/이송 조건 등) 변동 이력 확인'},
    'Edge-Ring': {'korean': '엣지 링 불량', 'cause': 'RMG', 'action': '웨이퍼 중심/에지 구간별 결함 분포 비교'},
    'Loc': {'korean': '국부 불량', 'cause': 'PC', 'action': 'PC 공정 전·후 표면 클리닝 상태 점검'},
    'Near-full': {'korean': '전면 불량', 'cause': '심각한 장비 고장, 원자재 불량', 'action': '즉시 생산 중단 및 장비 전수 점검'},
    'Random': {'korean': '랜덤 불량', 'cause': '정전기(ESD), 미세 스크래치', 'action': 'ESD 방지 대책 및 이송 환경 점검'},
    'Scratch': {'korean': '스크래치', 'cause': '물리적 접촉, 슬러리 이물질', 'action': '패드 상태, 압력 조건, 슬러리 공급 균일성 점검'}
}

# ==========================================
# 1-1. 결함유형 중심 도메인 설명 (공정 SHAP 빨간 표시 포함)
# ==========================================
DEFECT_DOMAIN_KB = {
    9: {"title": "CBCMP, PC, RMG – 9번 유형 (가성 불량 False)",
        "features": ('<span style="color:red">검출면적</span>, <span style="color:red">에너지값</span>'),
        "cause": ("• 검출면적이 클수록 강한 가성일 확률 높음<br>"
                  "• 에너지값이 클수록 강한 가성일 확률 높음"),
        "action": ("• 강한 가성인 경우 재검사 필요")},
    10: {"title": "RMG – 10번 유형(미세 파티클 Small Particle)",
         "features": ('<span style="color:red">기준편차</span>, 명도수준'),
         "cause": ("• 광학/센서 계측 불안정으로 인한 밝기 기반 결함<br>"
                   "• 기준편차 증가로 센서 드리프트 가능성<br>"
                   "• 실제 결함보다 장비 조건 영향 가능"),
         "action": ("• 조명·포커스·센서 캘리브레이션 점검<br>"
                    "• 재검을 통한 계측/실결함 구분<br>"
                    "• 기준편차·노이즈 트렌드 확인")},
    14: {"title": "PC – 14번 유형 (버블 Bubble)",
         "features": ('<span style="color:red">명도수준</span>, 영역잡음, 잡음정도'),
         "cause": ("• 국부 영역 잡음 집중, 파티클/잔사/오염 가능<br>"
                   "• 표면 반사 불균일로 명도 변화 증가<br>"
                   "• 세정 부족 또는 FOUP/이송 중 오염 가능"),
         "action": ("• 세정 조건 점검<br>"
                    "• 특정 로트·영역 집중 발생 확인<br>"
                    "• FOUP/보관 환경 점검")},
    17: {"title": "PC – 17번 유형(포토레지스트 잔여물 PR Residue)",
         "features": ('<span style="color:red">명도수준</span>, 잡음정도'),
         "cause": ("• 물리적 손상성 결함, 방향성 라인/스크래치 가능<br>"
                   "• 특정 방향 편중 패턴 발생 가능<br>"
                   "• 직전/PC 공정의 기계적 접촉 영향"),
         "action": ("• 롤러/가이드 등 접촉 부위 점검<br>"
                    "• 방향성 결함 패턴 확인<br>"
                    "• 장비 내부 이물 점검")},
    20: {"title": "RMG – 20번 유형(거대 파티클 Large Particle)",
         "features": ('<span style="color:red">기준편차</span>, 명도수준'),
         "cause": ("• 방향성 라인형/드래그성 결함<br>"
                   "• 이송/회전 방향 반복 자극 가능<br>"
                   "• 조건 변동으로 국부 과/부족 처리"),
         "action": ("• 결함 방향성과 장비 방향 비교<br>"
                    "• 기준편차 증가 구간 조건 점검<br>"
                    "• 타 로트 비교")},
    21: {"title": "CBCMP, RMG – 21번 유형 (금속 잔여물 Metal Residue)",
         "features": ('<span style="color:red">검출면적</span>, '
                      '<span style="color:red">정형지수</span>, '
                      '<span style="color:red">기준편차</span>, 명도수준'),
         "cause": ("• 강도·변동성 큰 에너지성 결함<br>"
                   "• CMP 압력/패드/슬러리 불균일 가능<br>"
                   "• 공정 안정성 저하로 국부 과/언더 발생"),
         "action": ("• 압력·패드·슬러리 균일성 점검<br>"
                    "• 레시피 변경/알람 시점 확인<br>"
                    "• 에지/센터 분포 분석")},
    22: {"title": "PC – 22번 유형(마이크로 스크래치 Micro-Scratch)",
         "features": ('<span style="color:red">명도수준</span>, 에너지값'),
         "cause": ("• 강한 국부 충돌/파손 이벤트 가능<br>"
                   "• 명암 변화와 함께 손상 패턴 발생<br>"
                   "• 이물 끼임 등 단발성 요인 가능"),
         "action": ("• 해당 웨이퍼 이력 점검<br>"
                    "• 장비 내부 이물 확인<br>"
                    "• 시간대별 생산 비교")},
    28: {"title": "PC – 28번 유형 (패턴 불량 Pattern Bridge)",
         "features": ('<span style="color:red">명도수준</span>, 정형지수'),
         "cause": ("• 형상 뚜렷한 패턴 결함<br>"
                   "• 방향성 긴 스크래치 가능<br>"
                   "• 명암·형상 특징 동시 강조"),
         "action": ("• 패턴 잔존/스크래치 여부 확인<br>"
                    "• 방향성 구조 이미지 분석<br>"
                    "• PC 조건 변화 시점 비교")},
    39: {"title": "CBCMP, RMG – 39번 유형",
         "features": ('<span style="color:red">정형지수</span>, '
                      '<span style="color:red">검출면적</span>, '
                      '<span style="color:red">기준편차</span>, 신호극성'),
         "cause": ("• 패턴성·반복 구조 결함<br>"
                   "• 포토/패턴 공정 영향 이월<br>"
                   "• 특정 패턴 반복 발생"),
         "action": ("• 포토/식각 이력 점검<br>"
                    "• 반복 패턴 여부 확인<br>"
                    "• 기준편차 높은 구간 분석")},
    56: {"title": "CBCMP – 56번 유형 (패드 자국 Pad Mark)",
         "features": ('<span style="color:red">검출면적</span>, '
                      '<span style="color:red">정형지수</span>, 명도수준'),
         "cause": ("• 명암 대비 큰 광학적 변화<br>"
                   "• 오염/산화막 편차 가능<br>"
                   "• CMP 균일도 저하"),
         "action": ("• 표면 산화/오염 점검<br>"
                    "• 패드 마모/압력 확인<br>"
                    "• 영역별 명도·결함 분포 확인")},
    99: {"title": "CBCMP – 99번 유형 (미분류 Unclassified)",
         "features": ('<span style="color:red">검출면적</span>, '
                      '<span style="color:red">정형지수</span>, 명도수준, 기준편차'),
         "cause": ("• 면적·형상·명암·변동성 복합 결함<br>"
                   "• 공정 변동성 증가 신호<br>"
                   "• 여러 요인 누적 가능"),
         "action": ("• CBCMP 전후 조건 이력 점검<br>"
                    "• 수율/명도/신호 트렌드 확인<br>"
                    "• 복합 유형으로 원인 세분화")}
}

# ==========================================
# 2. 모델 로딩 함수들
# ==========================================
@st.cache_resource
def load_real_fake_model():
    if not os.path.exists(MODEL_REAL_FAKE_PATH):
        return None, f"❌ REAL/FALSE 모델 파일 없음: {MODEL_REAL_FAKE_PATH}"
    try:
        with open(MODEL_REAL_FAKE_PATH, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"❌ REAL/FALSE 모델 로딩 오류: {e}"


@st.cache_resource
def load_defect_model():
    # ✅ joblib 로딩으로 변경
    if not os.path.exists(MODEL_DEFECT_PATH):
        return None, f"❌ 결함유형 모델 파일 없음: {MODEL_DEFECT_PATH}"
    try:
        obj = joblib.load(MODEL_DEFECT_PATH)

        # (1) dict로 저장된 경우 (예: {"model":..., "meta":...})
        if isinstance(obj, dict):
            for key in ["model", "clf", "classifier", "pipeline"]:
                if key in obj and hasattr(obj[key], "predict"):
                    return obj[key], None
            for v in obj.values():
                if hasattr(v, "predict"):
                    return v, None
            return None, "❌ best_defect_model.joblib 내부에서 predict 가능한 모델을 찾지 못했습니다."

        # (2) 바로 estimator / pipeline 인 경우
        if hasattr(obj, "predict"):
            return obj, None

        return None, "❌ best_defect_model.joblib 로딩은 되었지만 모델 객체가 아닙니다."
    except Exception as e:
        return None, f"❌ 결함유형 모델 로딩 오류: {e}"


# ==========================================
# 3. 스케일링 함수
# ==========================================
def robust_scale_single(input_df: pd.DataFrame, ref_df: pd.DataFrame, feature_cols):
    ref = ref_df[feature_cols].select_dtypes(include="number")
    med = ref.median()
    q1 = ref.quantile(0.25)
    q3 = ref.quantile(0.75)
    iqr = (q3 - q1).replace(0, 1.0)
    x = input_df[feature_cols].astype(float)
    return (x - med) / iqr


def log_robust_scale_single(input_df: pd.DataFrame, ref_df: pd.DataFrame,
                            feature_cols, log_cols):
    ref = ref_df[feature_cols].copy()
    for c in log_cols:
        if c in ref.columns:
            ref[c] = np.log1p(ref[c].clip(lower=0))

    med = ref.median()
    q1 = ref.quantile(0.25)
    q3 = ref.quantile(0.75)
    iqr = (q3 - q1).replace(0, 1.0)

    x = input_df[feature_cols].copy().astype(float)
    for c in log_cols:
        if c in x.columns:
            x[c] = np.log1p(x[c].clip(lower=0))

    return (x - med) / iqr


# ==========================================
# 4. REAL/FALSE 방향성 분석 (가성으로 가려면?)
# ==========================================
def compute_false_direction(input_df, df_final, model_rf, feature_cols):
    directions = {}
    try:
        if not hasattr(model_rf, "predict_proba"):
            return directions

        classes = getattr(model_rf, "classes_", np.array([0, 1]))
        if 0 in classes:
            idx_false = int(np.where(classes == 0)[0][0])
        else:
            return directions

        X_base = robust_scale_single(input_df, df_final, feature_cols)
        base_proba = model_rf.predict_proba(X_base)[0][idx_false]

        for f in feature_cols:
            val = float(input_df[f].iloc[0])
            abs_val = abs(val)
            step = max(abs_val * 0.1, 0.1) if abs_val != 0 else 1.0

            df_down = input_df.copy()
            df_up = input_df.copy()
            df_down[f] = val - step
            df_up[f] = val + step

            X_down = robust_scale_single(df_down, df_final, feature_cols)
            X_up = robust_scale_single(df_up, df_final, feature_cols)

            p_down = model_rf.predict_proba(X_down)[0][idx_false]
            p_up = model_rf.predict_proba(X_up)[0][idx_false]

            inc_down = p_down - base_proba
            inc_up = p_up - base_proba
            threshold = 0.01

            if (inc_down > threshold) and (inc_down > inc_up + 0.005):
                directions[f] = "down"
            elif (inc_up > threshold) and (inc_up > inc_down + 0.005):
                directions[f] = "up"
            else:
                directions[f] = "neutral"

    except Exception:
        return {}

    return directions


# ==========================================
# 5. 진성 확률 기반 공정 상태 라벨링
# ==========================================
def get_quality_status(prob_real: float):
    if prob_real is None:
        return "정보 부족", "진성 확률 정보 없음", "#7f8c8d", "⚪"

    p = float(prob_real)

    if p < 0.40:
        return "정상", "가성 결함 경향. 공정 이상 신호는 낮음.", "#27ae60", "🟢"
    elif p < 0.68:
        return "경고", "진성/가성 경계. 로트·장비 트렌드 점검 권장.", "#e67e22", "🟠"
    elif p < 0.95:
        return "불량", "진성 결함 가능성이 높은 영역.", "#e74c3c", "🔴"
    else:
        return "공정이상", "진성 결함 가능성이 매우 높음. 긴급점검 필요.", "#c0392b", "🚨"


# ==========================================
# 6. YOLO 멀티모달 모델 로딩
# ==========================================
@st.cache_resource
def load_multimodal_model():
    model = YOLO("best.pt")
    return model


def run_yolo_analysis(pil_image: Image.Image):
    model = load_multimodal_model()
    results = model.predict(source=pil_image, conf=0.25, save=False, verbose=False)
    result = results[0]
    annotated_frame = result.plot()

    detections = []
    if len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cname = CLASS_NAMES.get(cls_id, f"Class-{cls_id}")
            detections.append((cname, conf))

        main_defect = max(detections, key=lambda x: x[1])[0]
        knowledge = DEFECT_KNOWLEDGE_BASE.get(
            main_defect,
            {"korean": main_defect, "cause": "원인 미등록", "action": "조치 정보 없음"}
        )
    else:
        main_defect = None
        knowledge = None

    return annotated_frame, detections, main_defect, knowledge


# ==========================================
# 7. 페이지 본문 (main.py에서 호출)
# ==========================================
def show_page(df_final: pd.DataFrame):
    st.markdown("""
        <style>
            .card-header {
                background-color: #FFFFFF;
                border-top-left-radius: 20px;
                border-top-right-radius: 20px;
                border-bottom: 1px solid #ECECEC;
                padding: 14px 20px;
            }
            .card-title {
                font-size: 17px; font-weight: 700; color:#2d3436;
            }
            .card-body {
                background-color: #FFFFFF;
                border-bottom-left-radius: 20px;
                border-bottom-right-radius: 20px;
                padding: 16px 22px 20px 22px;
                box-shadow: 0 3px 12px rgba(0,0,0,0.05);
                margin-bottom: 22px;
            }
        </style>
    """, unsafe_allow_html=True)

    # -----------------------------
    # 모델 로딩
    # -----------------------------
    model_rf, err_rf = load_real_fake_model()
    model_defect, err_defect = load_defect_model()

    # 에러 메시지 출력(원하면 지워도 됨)
    if err_rf:
        st.error(err_rf)
    if err_defect:
        st.error(err_defect)

    st.markdown("<h2 style='font-weight:700;'>결함 예측 & 멀티모달 분석</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.5, 1.8, 1.3])

    # 세션 초기화
    if "pred_real_fake" not in st.session_state: st.session_state.pred_real_fake = None
    if "pred_real_conf" not in st.session_state: st.session_state.pred_real_conf = None
    if "pred_defect_type" not in st.session_state: st.session_state.pred_defect_type = None
    if "pred_defect_conf" not in st.session_state: st.session_state.pred_defect_conf = None
    if "last_input_df" not in st.session_state: st.session_state.last_input_df = None
    if "direction_hint" not in st.session_state: st.session_state.direction_hint = {}

    # ---------------------------------------------------------
    # (1) 왼쪽 열 — 피처 입력 + 예측 버튼
    # ---------------------------------------------------------
    with col1:
        st.markdown("<h4>① 입력 피처 설정</h4>", unsafe_allow_html=True)

        med = df_final[FEATURES].median(numeric_only=True)

        with st.form("input_form"):
            vals = {}
            cols = st.columns(2)

            for i, f in enumerate(FEATURES):
                c = cols[i % 2]
                with c:
                    vals[f] = st.number_input(f, value=float(med[f]), step=0.01, format="%.4f")

            submitted = st.form_submit_button("예측 실행", use_container_width=True)

        if submitted:
            try:
                input_df = pd.DataFrame([vals])[FEATURES]

                # -----------------
                # REAL/FALSE 예측
                # -----------------
                if model_rf is None:
                    raise RuntimeError("REAL/FALSE 모델이 로딩되지 않았습니다.")

                X_rf = robust_scale_single(input_df, df_final, FEATURES)

                classes_rf = getattr(model_rf, "classes_", np.array([0, 1]))
                if hasattr(model_rf, "predict_proba"):
                    prob_arr = np.array(model_rf.predict_proba(X_rf))[0]
                    if 1 in classes_rf:
                        idx_real = int(np.where(classes_rf == 1)[0][0])
                        prob_real = float(prob_arr[idx_real])
                    else:
                        prob_real = None
                else:
                    prob_real = None

                label_rf = "진성" if model_rf.predict(X_rf)[0] == 1 else "가성"

                st.session_state.pred_real_fake = label_rf
                st.session_state.pred_real_conf = prob_real
                st.session_state.last_input_df = input_df.copy()

                # -----------------
                # 결함유형 예측 (joblib)
                # -----------------
                if model_defect is not None:
                    X_def = log_robust_scale_single(input_df, df_final, FEATURES, LOG_FEATURES)

                    if hasattr(model_defect, "predict_proba"):
                        proba_def = np.array(model_defect.predict_proba(X_def))[0]
                        idx_max = int(np.argmax(proba_def))
                        st.session_state.pred_defect_type = map_defect_index(idx_max)
                        st.session_state.pred_defect_conf = float(proba_def[idx_max])
                    else:
                        raw = model_defect.predict(X_def)
                        raw = np.array(raw).flatten()[0]
                        st.session_state.pred_defect_type = map_defect_index(int(raw))
                        st.session_state.pred_defect_conf = None
                else:
                    st.session_state.pred_defect_type = None
                    st.session_state.pred_defect_conf = None

                # -----------------
                # 방향성 분석
                # -----------------
                st.session_state.direction_hint = compute_false_direction(
                    input_df, df_final, model_rf, FEATURES
                )

            except Exception as e:
                st.error(f"예측 오류: {e}")

    # ---------------------------------------------------------
    # (2) 가운데 열 — 예측결과 + 도메인 설명 (HTML 컴포넌트)
    # ---------------------------------------------------------
    with col2:
        st.markdown("<h4>② 머신러닝 결과 및 공정 상태</h4>", unsafe_allow_html=True)

        pred_rf = st.session_state.pred_real_fake
        prob_real = st.session_state.pred_real_conf
        pred_def = st.session_state.pred_defect_type
        pred_def_conf = st.session_state.pred_defect_conf

        if pred_rf is None and pred_def is None:
            st.info("좌측에서 피처 입력 후 **예측 실행**을 누르면 결과가 표시됩니다.")
        else:
            c1, c2 = st.columns(2)

            with c1:
                st.metric("진성/가성 예측", pred_rf)
                if prob_real is not None:
                    st.metric("진성 확률", f"{prob_real*100:.2f}%")

            with c2:
                if pred_def is not None:
                    st.metric("결함 유형", pred_def)
                    if pred_def_conf is not None:
                        st.metric("해당 코드 확률", f"{pred_def_conf*100:.2f}%")
                else:
                    st.metric("결함 유형", "모델 오류")

            quality_label, quality_desc, color_hex, icon = get_quality_status(prob_real)

            if pred_def in DEFECT_DOMAIN_KB:
                kb = DEFECT_DOMAIN_KB[pred_def]
                defect_html = f"""
                <div style="font-size:14px; line-height:1.7; color:#2d3436;">
                    <b>📌 {kb['title']}</b><br>
                    <b>주요 특징 피처:</b> {kb['features']}<br><br>
                    <b>가능한 원인</b><br>
                    {kb['cause']}<br><br>
                    <b>권장 조치</b><br>
                    {kb['action']}
                </div>
                """
            else:
                defect_html = """
                <div style="font-size:14px; color:#636e72;">
                    예측된 결함 코드에 대한 설명이 등록되지 않았습니다.
                </div>
                """

            components.html(
                f"""
                <div style="border-radius:18px; box-shadow:0 3px 12px rgba(0,0,0,0.06);
                            overflow:hidden; border:1px solid #EAEAEA;">

                    <div style="background:#FFF; padding:18px;
                                border-bottom:1px solid #F0F0F0; text-align:center;">
                        <div style="font-size:26px; font-weight:800; color:{color_hex};">
                            {icon} {quality_label}
                        </div>
                        <div style="margin-top:6px; font-size:13px; color:#636e72;">
                            {quality_desc}
                        </div>
                    </div>

                    <div style="background:#FFF; padding:18px 22px;">
                        {defect_html}
                    </div>

                </div>
                """,
                height=480,
                scrolling=True
            )

            if st.session_state.last_input_df is not None:
                with st.expander("입력값 다시 보기 (19개 FEATURES)", expanded=False):
                    st.dataframe(st.session_state.last_input_df, use_container_width=True)

                    if st.session_state.direction_hint:
                        st.markdown("#### 🔎 가성 방향성 힌트")
                        for f in FEATURES:
                            d = st.session_state.direction_hint.get(f, "neutral")
                            if d == "down":
                                st.markdown(f"- :blue[▼ {f}] : 값을 **낮추면** 가성확률↑")
                            elif d == "up":
                                st.markdown(f"- :red[▲ {f}] : 값을 **높이면** 가성확률↑")
                            else:
                                st.markdown(f"- {f} : 영향 미미(중립)")

    # ---------------------------------------------------------
    # (3) 오른쪽 열 — 이미지 기반 형상 분류 (YOLO)
    # ---------------------------------------------------------
    with col3:
        st.markdown("<h4>③ 이미지 기반 형상 분석 (YOLO)</h4>", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "웨이퍼 결함 이미지 업로드 (png/jpg/jpeg)",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded is not None:
            try:
                image = Image.open(uploaded)

                annotated, det_list, main_def, know = run_yolo_analysis(image)

                st.markdown("#### 업로드 이미지")
                st.image(image, use_container_width=True)

                st.markdown("#### YOLO 형상 분석 결과")

                if main_def is None:
                    st.success("📌 YOLO 모델이 결함 박스를 검출하지 못했습니다. (정상 또는 경미 결함)")
                else:
                    st.markdown(f"**주 결함 유형:** {know['korean']} ({main_def})")
                    st.markdown(f"**공정명:** {know['cause']}")
                    st.markdown(f"**권장 조치:** {know['action']}")

                if det_list:
                    st.markdown("#### 검출된 결함 박스 목록")
                    det_df = pd.DataFrame(det_list, columns=["불량유형(영문)", "신뢰도(conf)"])
                    det_df["신뢰도(%)"] = (det_df["신뢰도(conf)"] * 100).round(2)
                    det_df = det_df.sort_values("신뢰도(conf)", ascending=False)
                    st.dataframe(det_df[["불량유형(영문)", "신뢰도(%)"]], use_container_width=True)

                st.markdown("#### YOLO 출력 이미지")
                st.image(annotated, channels="BGR", use_container_width=True)

            except Exception as e:
                st.error(f"YOLO 분석 중 오류 발생: {e}")
                st.info("· best.pt 파일 경로 또는 모델 버전이 정확한지 확인하세요.")
        else:
            st.info("YOLO 분석을 위해 결함 이미지를 업로드하세요.")


# ==========================================
# (END OF FILE)
# ==========================================
