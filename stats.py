import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------------------------
# 1) Plotly SPC 관리도 함수
# --------------------------------------------------------------------------
def make_spc_chart_plotly(df_src: pd.DataFrame, var: str):
    if ('배치번호' not in df_src.columns) or (var not in df_src.columns):
        return None

    df_spc = df_src[['배치번호', var]].dropna().copy()
    if df_spc.empty:
        return None

    batch_order = df_spc['배치번호'].unique()
    batch_map = {b: idx for idx, b in enumerate(batch_order)}
    df_spc['Batch_Index'] = df_spc['배치번호'].map(batch_map)

    batch_avg = (
        df_spc.groupby('Batch_Index')[var]
              .mean()
              .reset_index()
              .sort_values('Batch_Index')
    )

    x = batch_avg['Batch_Index']
    y = batch_avg[var]

    mean = y.mean()
    std = y.std()
    ucl = mean + 3 * std
    lcl = mean - 3 * std

    out_mask = (y > ucl) | (y < lcl)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines+markers',
        marker=dict(size=6, color="#6C5CE7"),
        line=dict(color="#6C5CE7", width=2),
        name="Batch Mean"
    ))

    fig.add_trace(go.Scatter(
        x=x[out_mask], y=y[out_mask],
        mode='markers',
        marker=dict(size=10, color='red'),
        name="Out of Limit"
    ))

    fig.add_hline(y=mean, line_color="green", annotation_text=f"CL {mean:.2f}")
    fig.add_hline(y=ucl, line_color="red", line_dash="dash", annotation_text=f"UCL {ucl:.2f}")
    fig.add_hline(y=lcl, line_color="red", line_dash="dash", annotation_text=f"LCL {lcl:.2f}")

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white",
        showlegend=False,
    )

    return fig


# ==============================================================================
#                                 show_page(df)
# ==============================================================================
def show_page(df: pd.DataFrame):

    # 헤더
    st.markdown("""
        <div style="font-size:28px; font-weight:700; color:#2D3436;">
            공정결함 통계요약
        </div>
        <div style="color:#888; margin-bottom:25px;">
            SPC 기반 공정관리 / 변수별 Cpk 분석 / 불량요인 파악
        </div>
    """, unsafe_allow_html=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # 불량 라벨 생성
    if '불량여부_le' in df.columns:
        df['_불량라벨'] = df['불량여부_le'].apply(lambda x: '불량' if x == 1 else '정상')
    else:
        df['_불량라벨'] = df['불량여부'].astype(str).apply(
            lambda v: '불량' if v.upper() not in ['0', 'FALSE'] else '정상'
        )

    # 공정 × 결함 조합
    if '공정명' in df.columns and '결함유형' in df.columns:
        combo_counts = (
            df.groupby(['공정명', '결함유형'])
              .size()
              .reset_index(name='Count')
              .sort_values('Count', ascending=False)
        )
    else:
        combo_counts = None

    # ----------------------------------------------------------------------
    # 상단 compact 필터 (SPC 그룹)
    # ----------------------------------------------------------------------
    spc_groups = {
        "에너지/물리 결함": ["에너지값", "검출면적"],
        "신호/잡음 결함": ["신호강도", "잡음정도"],
        "SHAP 기준 결함": ["명도수준", "기준편차"]
    }

    st.markdown("""
        <div style='display:flex; justify-content:flex-end; margin-bottom:-10px;'>
            <span style="font-size:13px; color:#6C5CE7; font-weight:600; margin-right:6px;">
                SPC 변수 그룹:
            </span>
        </div>
    """, unsafe_allow_html=True)

    selected_group = st.selectbox("", list(spc_groups.keys()), key="spc_select_top")

    var_left, var_mid = spc_groups[selected_group]

    # ----------------------------------------------------------------------
    # 상단 섹션 : SPC 2개 + Cpk 순위
    # ----------------------------------------------------------------------
    col_left, col_mid, col_right = st.columns([2, 2, 1])

    # 왼쪽 SPC
    with col_left:
        st.markdown(f"<h5>{var_left}</h5>", unsafe_allow_html=True)
        fig1 = make_spc_chart_plotly(df, var_left)
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info(f"{var_left} 관리도를 그릴 수 없습니다.")

    # 가운데 SPC
    with col_mid:
        st.markdown(f"<h5>{var_mid}</h5>", unsafe_allow_html=True)
        fig2 = make_spc_chart_plotly(df, var_mid)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info(f"{var_mid} 관리도를 그릴 수 없습니다.")

    # Cpk 계산 함수 및 등급
    spec = {
        "에너지값": (0, 6000),
        "검출면적": (0, 0.5),
        "신호강도": (0, 1500),
        "잡음정도": (0, 800),
        "명도수준": (0, 500),
        "기준편차": (0, 300)
    }

    def compute_cpk(series, lsl, usl):
        s = series.dropna()
        if len(s) < 3:
            return np.nan
        m, sd = s.mean(), s.std()
        if sd == 0:
            return np.nan
        cpu = (usl - m) / (3 * sd)
        cpl = (m - lsl) / (3 * sd)
        return min(cpu, cpl)

    def cpk_status(cpk):
        if cpk >= 1.67: return "최우수 (6σ)", "#6C5CE7"
        elif cpk >= 1.33: return "우수 (1등급)", "#0984e3"
        elif cpk >= 1.0:  return "양호 (2등급)", "#00b894"
        elif cpk >= 0.67: return "미흡 (3등급)", "#fdcb6e"
        else:             return "불량 (관리필요)", "#d63031"

    # 오른쪽 Cpk 리스트
    with col_right:
        st.markdown("<h5>Cpk 순위</h5>", unsafe_allow_html=True)

        rows = []
        for v in spc_groups["에너지/물리 결함"] + spc_groups["신호/잡음 결함"] + spc_groups["SHAP 기준 결함"]:
            if v in df.columns and v in spec:
                lsl, usl = spec[v]
                cpk = compute_cpk(df[v], lsl, usl)
                if not np.isnan(cpk):
                    rows.append((v, cpk))

        if rows:
            cpk_df = pd.DataFrame(rows, columns=["변수", "Cpk"]).sort_values("Cpk", ascending=False)

            html = "<ul style='font-size:13px; line-height:1.6;'>"
            for _, row in cpk_df.iterrows():
                status, color = cpk_status(row["Cpk"])
                html += (
                    f"<li><b>{row['변수']}</b> : {row['Cpk']:.3f} "
                    f"&rarr; <span style='color:{color}; font-weight:600;'>{status}</span></li>"
                )
            html += "</ul>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("Cpk를 계산할 수 있는 변수가 없습니다.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # 🔻 중단 섹션 : 히스토그램 + Six Sigma(Batch_Index) + 이상치 Top10
    # ----------------------------------------------------------------------
    mid_features = [
        '가로길이', '세로길이', '검출면적', '직경크기',
        '신호강도', '신호극성', '에너지값', '기준편차',
        '명도수준', '잡음정도',
        '정렬정도', '점형지수', '영역잡음', '상대강도',
        '활성지수', '패치신호'
    ]
    mid_features = [c for c in mid_features if c in df.columns]

    # Batch_Index 없으면 생성
    if '배치번호' in df.columns and 'Batch_Index' not in df.columns:
        batch_order_mid = df['배치번호'].unique()
        batch_map_mid = {b: idx for idx, b in enumerate(batch_order_mid)}
        df['Batch_Index'] = df['배치번호'].map(batch_map_mid)

    if mid_features and 'Batch_Index' in df.columns:
        # compact 필터
        st.markdown("""
            <div style='display:flex; justify-content:flex-end; margin-top:5px; margin-bottom:-10px;'>
                <span style="font-size:13px; color:#6C5CE7; font-weight:600; margin-right:6px;">
                    분석 변수 선택:
                </span>
            </div>
        """, unsafe_allow_html=True)

        selected_mid_feature = st.selectbox(
            "",
            mid_features,
            key="mid_feature_select"
        )

        series = df[selected_mid_feature].dropna()
        if len(series) > 1:
            μ_raw = series.mean()
            σ_raw = series.std()
        else:
            μ_raw, σ_raw = series.mean(), 0.0

        # 레이아웃
        m_left, m_mid, m_right = st.columns([2, 2, 1])

        # ----------------- Left : 히스토그램 + 정규분포 -----------------
        with m_left:
            st.markdown(f"<h5>{selected_mid_feature} 분포</h5>", unsafe_allow_html=True)

            if len(series) > 1:
                bins = 40
                counts, bin_edges = np.histogram(series, bins=bins)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                if σ_raw > 0:
                    pdf = (1 / (σ_raw * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((bin_centers - μ_raw) / σ_raw) ** 2)
                    bin_width = bin_edges[1] - bin_edges[0]
                    pdf_scaled = pdf * len(series) * bin_width
                else:
                    pdf_scaled = np.zeros_like(bin_centers)

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Bar(
                    x=bin_centers,
                    y=counts,
                    name="Count",
                    marker_color="#6C5CE7",
                    opacity=0.75
                ))
                fig_hist.add_trace(go.Scatter(
                    x=bin_centers,
                    y=pdf_scaled,
                    mode="lines",
                    name="Normal PDF",
                    line=dict(color="#E17055", width=2)
                ))
                fig_hist.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=40, b=10),
                    plot_bgcolor="white",
                    xaxis_title=selected_mid_feature,
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info(f"{selected_mid_feature} 값이 너무 적어 분포를 그릴 수 없습니다.")

        # ----------------- Middle : Six Sigma (Batch_Index) -----------------
        with m_mid:
            st.markdown(f"<h5>{selected_mid_feature} Six-Sigma</h5>", unsafe_allow_html=True)

            df_six = df[['Batch_Index', selected_mid_feature]].dropna()
            if not df_six.empty:
                df_six = (
                    df_six.groupby('Batch_Index')[selected_mid_feature]
                          .mean()
                          .reset_index()
                          .sort_values('Batch_Index')
                )

                y = df_six[selected_mid_feature]
                x = df_six['Batch_Index']

                μ_batch = y.mean()
                σ_batch = y.std()

                if σ_batch > 0:
                    z1p = μ_batch + 1 * σ_batch
                    z2p = μ_batch + 2 * σ_batch
                    z3p = μ_batch + 3 * σ_batch
                    z1n = μ_batch - 1 * σ_batch
                    z2n = μ_batch - 2 * σ_batch
                    z3n = μ_batch - 3 * σ_batch
                else:
                    z1p = z2p = z3p = μ_batch
                    z1n = z2n = z3n = μ_batch

                fig_six = go.Figure()

                if σ_batch > 0:
                    # Zone C (±1σ) - 초록
                    fig_six.add_hrect(
                        y0=z1n, y1=z1p,
                        fillcolor="#C8E6C9", opacity=0.6, line_width=0,
                        layer="below"
                    )
                    # Zone B (1~2σ) - 노랑
                    fig_six.add_hrect(
                        y0=z2n, y1=z1n,
                        fillcolor="#FFF9C4", opacity=0.6, line_width=0,
                        layer="below"
                    )
                    fig_six.add_hrect(
                        y0=z1p, y1=z2p,
                        fillcolor="#FFF9C4", opacity=0.6, line_width=0,
                        layer="below"
                    )
                    # Zone A (2~3σ) - 빨강
                    fig_six.add_hrect(
                        y0=z3n, y1=z2n,
                        fillcolor="#FFCDD2", opacity=0.6, line_width=0,
                        layer="below"
                    )
                    fig_six.add_hrect(
                        y0=z2p, y1=z3p,
                        fillcolor="#FFCDD2", opacity=0.6, line_width=0,
                        layer="below"
                    )

                # 보라색 라인 + 점 (SPC와 동일 톤)
                fig_six.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    line=dict(color="#6C5CE7", width=2),
                    marker=dict(size=6, color="#6C5CE7"),
                    name="Batch Mean"
                ))

                # ±3σ 넘은 점만 빨간 점
                if σ_batch > 0:
                    mask_out = (y > z3p) | (y < z3n)
                    fig_six.add_trace(go.Scatter(
                        x=x[mask_out],
                        y=y[mask_out],
                        mode="markers",
                        marker=dict(size=8, color="#d63031"),
                        name="Out of ±3σ"
                    ))

                # 기준선 표시
                fig_six.add_hline(y=μ_batch, line_color="#2ecc71", annotation_text=f"Mean {μ_batch:.2f}")
                fig_six.add_hline(y=z1p, line_color="#95a5a6", line_dash="dot", annotation_text="+1σ")
                fig_six.add_hline(y=z1n, line_color="#95a5a6", line_dash="dot", annotation_text="-1σ")
                fig_six.add_hline(y=z2p, line_color="#f1c40f", line_dash="dot", annotation_text="+2σ")
                fig_six.add_hline(y=z2n, line_color="#f1c40f", line_dash="dot", annotation_text="-2σ")
                fig_six.add_hline(y=z3p, line_color="#e74c3c", line_dash="dash", annotation_text="+3σ")
                fig_six.add_hline(y=z3n, line_color="#e74c3c", line_dash="dash", annotation_text="-3σ")

                fig_six.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=40, b=10),
                    plot_bgcolor="white",
                    xaxis_title="Batch_Index",
                    yaxis_title=selected_mid_feature,
                    showlegend=False
                )

                st.plotly_chart(fig_six, use_container_width=True)
            else:
                st.info("Batch 기준 데이터를 생성할 수 없습니다.")

        # ----------------- Right : 이상치 Top10 -----------------
        with m_right:
            st.markdown("<h5>이상치 Top10</h5>", unsafe_allow_html=True)

            outlier_summary = []
            for col in mid_features:
                s = df[col].dropna()
                if len(s) < 2:
                    outlier_summary.append((col, 0))
                    continue
                m = s.mean()
                sd = s.std()
                upper = m + 3 * sd
                lower = m - 3 * sd
                cnt = ((s > upper) | (s < lower)).sum()
                outlier_summary.append((col, cnt))

            outlier_summary = sorted(outlier_summary, key=lambda x: x[1], reverse=True)[:10]

            html = "<ul style='font-size:13px; line-height:1.6;'>"
            for idx, (col, oc) in enumerate(outlier_summary):
                if idx < 3:
                    html += f"<li style='color:#d63031; font-weight:700;'>🔴 {col} : {oc}건</li>"
                else:
                    html += f"<li>{col} : {oc}건</li>"
            html += "</ul>"

            st.markdown(html, unsafe_allow_html=True)

    else:
        st.info("중단 섹션에 사용할 수 있는 수치형 컬럼 혹은 Batch_Index가 없습니다.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # 🔻 마지막 섹션 : 숫자형 기술통계
    # ----------------------------------------------------------------------
    st.markdown("<h5>숫자형 기술통계</h5>", unsafe_allow_html=True)

    if num_cols:
        desc = df[num_cols].describe().T
        st.dataframe(desc, use_container_width=True)
    else:
        st.info("숫자형 변수가 없습니다.")
