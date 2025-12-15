import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.ndimage import gaussian_filter
import os
import pickle

# ------------------------------------------------------
# 0. REAL/FALSE LGBM 모델 설정
# ------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_REAL_FAKE_PATH = os.path.join(current_dir, "lgbm_v4.pkl")

# 학습에 사용했던 피처 목록
FEATURES = [
    '가로길이', '세로길이', '검출면적', '직경크기', '신호강도', '신호극성',
    '에너지값', '기준편차', '명도수준', '잡음정도', '중심거리', '방향각도',
    '정렬정도', '점형지수', '영역잡음', '상대강도', '활성지수', '패치신호', 'Aspect_Ratio'
]


@st.cache_resource
def load_real_fake_model():
    """lgbm_v4.pkl 로드 (REAL/FALSE 분류용)"""
    if not os.path.exists(MODEL_REAL_FAKE_PATH):
        return None, f"❌ REAL/FALSE 모델 파일 없음: {MODEL_REAL_FAKE_PATH}"
    try:
        with open(MODEL_REAL_FAKE_PATH, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"❌ REAL/FALSE 모델 로딩 오류: {e}"


def robust_scale_for_kpi(df: pd.DataFrame, feature_cols):
    """
    KPI용 로버스트 스케일링 (df 전체를 기준으로 median / IQR 계산)
    """
    ref = df[feature_cols].select_dtypes(include="number")
    med = ref.median()
    q1 = ref.quantile(0.25)
    q3 = ref.quantile(0.75)
    iqr = (q3 - q1).replace(0, 1.0)
    x = ref.astype(float)
    return (x - med) / iqr


def show_page(df):
    if df.empty:
        st.warning("데이터가 존재하지 않습니다.")
        return

    # ------------------------------------------------------------------
    # [스타일 정의]
    # ------------------------------------------------------------------
    st.markdown("""
        <style>
            .card-header {
                background-color: #FFFFFF;
                border-top-left-radius: 20px;
                border-top-right-radius: 20px;
                border-bottom: 1px solid #F0F0F0;
                padding: 20px 24px 10px 24px;
            }
            .card-title {
                color: #2D3436;
                font-size: 18px;
                font-weight: 700;
                margin: 0;
            }
            .card-body {
                background-color: #FFFFFF;
                border-bottom-left-radius: 20px;
                border-bottom-right-radius: 20px;
                padding: 0px 20px 20px 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.03);
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # [기본값 설정] Raw View (블러 끄기)
    # ------------------------------------------------------------------
    if 'use_blur' not in st.session_state:
        st.session_state['use_blur'] = False

    def toggle_blur():
        st.session_state['use_blur'] = not st.session_state['use_blur']

    # ------------------------------------------------------------------
    # [드릴다운 레벨 감지 로직]
    # ------------------------------------------------------------------
    unique_procs = df['공정명'].nunique()
    unique_types = df['결함유형'].nunique()

    if unique_procs > 1:
        current_scope = "전체 공정"
        group_col = '공정명'
        color_col = '공정명'
        bar_title = "공정별 집계"
    elif unique_types > 1:
        current_scope = df['공정명'].iloc[0]
        group_col = '결함유형'
        color_col = '결함유형'
        bar_title = "결함 유형별 집계"
    else:
        current_scope = f"{df['결함유형'].iloc[0]}"
        group_col = '배치번호'
        color_col = '배치번호'
        bar_title = "배치별 집계"

    # ------------------------------------------------------------------
    # 1. Header
    # ------------------------------------------------------------------
    st.markdown(
        '<div class="header-text" style="font-size:26px; font-weight:700;"> 생산라인 요약 리포트</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header-text" style="font-size:14px; color:#636E72;">현재 생산 라인의 주요 지표를 요약해드립니다.</div>',
        unsafe_allow_html=True
    )

    # ------------------------------------------------------------------
    # 2. KPI Cards
    # ------------------------------------------------------------------
    total_wafers = len(df)

    defect_df = df[df['불량여부'].astype(str).str.upper().isin(['REAL', '1', 'TRUE', 'DEFECT'])]
    defect_count = len(defect_df)

    defect_rate = (defect_count / total_wafers) * 100 if total_wafers > 0 else 0
    yield_rate = 100 - defect_rate

    if 'defect_count' in df.columns:
        avg_defects = df['defect_count'].mean()
    else:
        avg_defects = defect_count / total_wafers if total_wafers > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(" 총 웨이퍼 수", f"{total_wafers:,}", "건수")
    c2.metric(" 수율(Yield)", f"{yield_rate:.1f}%", "비율")
    c3.metric(" 불량률", f"{defect_rate:.1f}%", "비율", delta_color="inverse")
    c4.metric(" 평균 불량 수", f"{avg_defects:.2f}", "웨이퍼당", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 3. Charts Section
    # ------------------------------------------------------------------
    col_left, col_center, col_right = st.columns([1, 1.5, 1.5])

    # --- [Left] Defect Ratio Gauge ---
    with col_left:
        gauge_title = f"{current_scope} 불량률"
        st.markdown(
            f'<div class="card-header"><h5 class="card-title"> {gauge_title}</h5></div><div class="card-body">',
            unsafe_allow_html=True
        )

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=defect_rate,
            number={'suffix': "%", 'font': {'color': "#6C5CE7"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#6C5CE7"},
                'bgcolor': "white",
                'steps': [{'range': [0, 100], 'color': "#ECEBFF"}],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            height=220,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(
            f"<div style='text-align:center; color:#636E72; font-size:12px;'>불량 {defect_count:,}건 / 총 {total_wafers:,}건</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # --- [Center] Process Trends ---
    with col_center:
        st.markdown(
            f'<div class="card-header"><h5 class="card-title"> {bar_title} 추세</h5></div><div class="card-body">',
            unsafe_allow_html=True
        )

        if group_col in df.columns:
            chart_stats = df.groupby(group_col).size().reset_index(name='Count')
            chart_stats[group_col] = chart_stats[group_col].astype(str)
            chart_stats = chart_stats.sort_values(by='Count', ascending=True)

            fig_bar = px.bar(
                chart_stats,
                x='Count',
                y=group_col,
                orientation='h',
                text='Count',
                color='Count',
                color_continuous_scale=[(0, "#ECEBFF"), (1, "#6C5CE7")]
            )

            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=10, l=0, r=0, b=10),
                height=260,
                xaxis=dict(showgrid=True, gridcolor='#F0F0F0'),
                yaxis=dict(showgrid=False, type='category'),
                barcornerradius=5,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info(f"({group_col}) 컬럼이 존재하지 않아 차트를 표시할 수 없습니다.")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- [Right] Wafer Map ---
    with col_right:
        btn_text = " View " if not st.session_state['use_blur'] else " Gaussian Blur "
        st.markdown(f'''
            <div class="card-header" style="display:flex; justify-content:space-between;">
                <h5 class="card-title"> 웨이퍼 맵</h5>
            </div>
            <div class="card-body">
        ''', unsafe_allow_html=True)

        if st.button(btn_text + "", key='blur_toggle', use_container_width=True):
            toggle_blur()
            st.rerun()

        if 'wafer_x' in df.columns and 'wafer_y' in df.columns:
            if st.session_state['use_blur']:
                # Blur mode
                try:
                    heatmap, xedges, yedges = np.histogram2d(
                        df['wafer_x'], df['wafer_y'], bins=100
                    )
                    heatmap_blurred = gaussian_filter(heatmap, sigma=4)
                    fig_map = go.Figure(data=go.Heatmap(
                        z=heatmap_blurred.T,
                        colorscale='Plasma',
                        showscale=False
                    ))
                except:
                    st.error("좌표 변환 중 오류가 발생했습니다.")
                    fig_map = go.Figure()
            else:
                # Raw view
                plot_df = df.copy()
                plot_df[color_col] = plot_df[color_col].astype(str)

                custom_palette = [
                    '#6C5CE7', '#A29BFE', '#74B9FF', '#0984E3',
                    '#00CEC9', '#81ECEC', '#FD79A8', '#E84393'
                ]

                fig_map = px.scatter(
                    plot_df,
                    x='wafer_x',
                    y='wafer_y',
                    color=color_col,
                    opacity=0.8,
                    color_discrete_sequence=custom_palette
                )
                fig_map.update_traces(marker=dict(size=2))

                if plot_df[color_col].nunique() > 10:
                    fig_map.update_layout(showlegend=False)

            fig_map.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=220,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1)
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("좌표 데이터(wafer_x, wafer_y)가 존재하지 않습니다.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================================
    #  실시간 예측 결과 알람 리포트 (Dashboard Bottom)
    # ======================================================================

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="card-header"><h5 class="card-title"> 실시간 예측 결과 알람 리포트</h5></div>'
        '<div class="card-body">',
        unsafe_allow_html=True
    )

    # 1) 모델 로드
    model, model_err = load_real_fake_model()
    if model_err:
        st.error(model_err)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # 2) 피처 체크
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"❌ 알람 예측에 필요한 피처가 없습니다: {missing}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # 3) 예측 확률 생성 (불량(REAL) 확률)
    try:
        X_scaled = robust_scale_for_kpi(df, FEATURES)
        proba = model.predict_proba(X_scaled)
        y_pred_prob = proba[:, 1]
    except Exception as e:
        st.error(f"❌ 예측 중 오류가 발생했습니다: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # 4) 임계값 정의
    threshold_warning = 0.4000
    threshold_defect  = 0.6826
    threshold_anomaly = 0.9546

    # 5) 구간별 샘플 분류 (우선순위: 공정이상 > 불량 > 경고 > 정상)
    anomaly_indices = np.where(y_pred_prob >= threshold_anomaly)[0]
    defect_indices  = np.where(
        (y_pred_prob >= threshold_defect) &
        (y_pred_prob < threshold_anomaly)
    )[0]
    warning_indices = np.where(
        (y_pred_prob >= threshold_warning) &
        (y_pred_prob < threshold_defect)
    )[0]
    normal_indices  = np.where(y_pred_prob < threshold_warning)[0]

    # 6) 요약 메트릭
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚨 공정이상", f"{len(anomaly_indices):,}건")
    c2.metric("🔴 불량", f"{len(defect_indices):,}건")
    c3.metric("🟠 경고", f"{len(warning_indices):,}건")
    c4.metric("🟢 정상", f"{len(normal_indices):,}건")

    # 7) 상세(상위 5개)
    def _show_top(indices, title, emoji, max_rows=5):
        if len(indices) == 0:
            st.success(f"{emoji} {title}: 없음")
            return

        show_cols = [
            c for c in
            ["공정명", "배치번호", "웨이퍼위치", "검사순번", "결함유형", "불량여부"]
            if c in df.columns
        ]

        rows = []
        for i in indices[:max_rows]:
            row = {c: df.iloc[i][c] for c in show_cols}
            row["샘플인덱스"] = int(i)
            row["예측확률"] = float(y_pred_prob[i])
            rows.append(row)

        out = pd.DataFrame(rows).sort_values("예측확률", ascending=False)
        st.dataframe(out, use_container_width=True)

    with st.expander("상세 알람 보기 (클릭)"):
        # 1. CRITICAL (공정이상)
        if len(anomaly_indices) > 0:
            st.markdown(f"##### 🚨 CRITICAL 공정이상 (>= {threshold_anomaly})")
        _show_top(
            anomaly_indices,
            f"CRITICAL 공정이상 (>= {threshold_anomaly})",
            "🚨"
        )

        # 2. DEFECT (불량 의심)
        if len(defect_indices) > 0:
            st.markdown(f"##### 🔴 DEFECT 불량 의심 ({threshold_defect} ~ {threshold_anomaly})")
        _show_top(
            defect_indices,
            f"DEFECT 불량 의심 ({threshold_defect} ~ {threshold_anomaly})",
            "🔴"
        )

        # 3. WARNING (경고)
        if len(warning_indices) > 0:
            st.markdown(f"##### 🟠 WARNING 경고 ({threshold_warning} ~ {threshold_defect})")
        _show_top(
            warning_indices,
            f"WARNING 경고 ({threshold_warning} ~ {threshold_defect})",
            "🟠"
        )

    st.markdown("</div>", unsafe_allow_html=True)

