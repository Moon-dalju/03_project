import streamlit as st
import pandas as pd
import os

# --------------------------------------------------------------------------------
# 1. 페이지 기본 설정
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="반도체 분석",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------------
# 2. 커스텀 CSS
# --------------------------------------------------------------------------------
st.markdown("""
    <style>
        .stApp { background-color: #F8F9FD; font-family: 'Inter', 'Suit', sans-serif; }
        [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E0E0; }
        [data-testid="stSidebar"] h1 { color: #6C5CE7; font-size: 24px; font-weight: 800; padding-left: 10px; }

        .stRadio > div[role="radiogroup"] > label > div:first-child { display: none !important; }
        .stRadio > div[role="radiogroup"] > label {
            background-color: transparent; border: none; padding: 12px 20px !important;
            margin-bottom: 8px !important; border-radius: 12px !important;
            color: #636e72; font-weight: 600; cursor: pointer; transition: all 0.2s; width: 100%;
        }
        .stRadio > div[role="radiogroup"] > label:hover { background-color: #F1F3F6 !important; color: #6C5CE7 !important; }
        .stRadio > div[role="radiogroup"] > label[data-checked="true"] {
            background-color: #6C5CE7 !important; color: white !important;
            box-shadow: 0 4px 10px rgba(108, 92, 231, 0.4);
        }

        .card-header { background-color: #FFFFFF; border-top-left-radius: 20px; border-top-right-radius: 20px; border-bottom: 1px solid #F0F0F0; padding: 20px 24px 10px 24px; }
        .card-body { background-color: #FFFFFF; border-bottom-left-radius: 20px; border-bottom-right-radius: 20px; padding: 0px 20px 20px 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); margin-bottom: 20px; }
        [data-testid="stMetric"] { background-color: #FFFFFF; border-radius: 16px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.02); border: 1px solid #F5F5F5; }
        div[data-baseweb="select"] > div { background-color: #F8F9FA; border-radius: 12px; border: 1px solid #E6E6E6; }
    </style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------
# 3. 데이터 소스 설정
# --------------------------------------------------------------------------------
DATA_SOURCE = os.getenv("DATA_SOURCE", "csv").lower()


# --------------------------------------------------------------------------------
# 4. 데이터 로드 함수
# --------------------------------------------------------------------------------
@st.cache_data
def load_data(data_source: str):
    df = None
    is_realtime = False

    if data_source == "db":
        pass

    elif data_source == "api":
        pass

    # CSV fallback
    if df is None:
        file_names = [
            'cleaned_wafer_data.csv',
            '반도체_결함_데이터_한글.csv'
        ]
        for fpath in file_names:
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                break
        is_realtime = False

    # 공통 전처리
    if df is not None:
        col_map = {
            'Process': '공정명', 'process': '공정명',
            'failureType': '결함유형', 'defect_type': '결함유형',
            'lotName': '배치번호', 'batch_no': '배치번호',
            'x': 'wafer_x', 'y': 'wafer_y',
            'is_defect': '불량여부', 'label': '불량여부'
        }
        df.rename(columns=col_map, inplace=True)

        if '공정명' not in df.columns: df['공정명'] = 'Unknown'
        if '결함유형' not in df.columns: df['결함유형'] = 'Normal'
        if '배치번호' not in df.columns: df['배치번호'] = 'Batch_001'

        for col in ['공정명', '결함유형', '배치번호']:
            df[col] = df[col].astype(str)

        if '불량여부' not in df.columns:
            df['불량여부'] = df['결함유형'].apply(
                lambda x: 'NORMAL' if str(x).lower() in ['none', 'normal', 'nan'] else 'REAL'
            )

    return df, is_realtime


df_raw, REALTIME_ACTIVE = load_data(DATA_SOURCE)


# --------------------------------------------------------------------------------
# 5. 사이드바
# --------------------------------------------------------------------------------
with st.sidebar:

    # 💜 원래대로 롤백된 제목
    st.title("🟣 반도체")
    st.markdown("<br>", unsafe_allow_html=True)

    menu = st.radio("Menu", ["Dashboard", "Stats", "Machine"], label_visibility="collapsed")
    st.subheader("Filter")

    # 필터 처리
    if df_raw is not None:
        proc_opts = ["전체"] + sorted(df_raw['공정명'].unique().tolist())
        sel_proc = st.selectbox("공정명 (Process)", proc_opts)
        df1 = df_raw if sel_proc == "전체" else df_raw[df_raw['공정명'] == sel_proc]

        defect_opts = ["전체"] + sorted(df1['결함유형'].unique().tolist())
        sel_defect = st.selectbox("결함유형 (Type)", defect_opts)
        df2 = df1 if sel_defect == "전체" else df1[df1['결함유형'] == sel_defect]

        batch_opts = ["전체"] + sorted(df2['배치번호'].unique().tolist())
        sel_batch = st.selectbox("배치번호 (Batch)", batch_opts)
        df_final = df2 if sel_batch == "전체" else df2[df2['배치번호'] == sel_batch]

        st.markdown(
            f"<div style='text-align:right; color:#888; font-size:12px;'>선택 데이터: {len(df_final):,} 건</div>",
            unsafe_allow_html=True
        )
    else:
        df_final = pd.DataFrame()
        st.error("데이터 로드 실패")

    st.markdown("<hr>", unsafe_allow_html=True)

    if REALTIME_ACTIVE:
        st.markdown("<div style='text-align:center; color:#27AE60; font-weight:700;'>● 작동중</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center; color:#E74C3C; font-weight:700;'>● 중단</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------------
# 6. 페이지 라우팅
# --------------------------------------------------------------------------------
if not df_final.empty:

    if menu == "Dashboard":
        try:
            import KPI
            KPI.show_page(df_final)
        except Exception as e:
            st.error(f"KPI.py 오류: {e}")

    elif menu == "Stats":
        try:
            import stats
            stats.show_page(df_final)
        except:
            st.info("stats.py 파일 없음")

    elif menu == "Machine":
        try:
            import machine
            machine.show_page(df_final)
        except Exception as e:
            st.error(f"machine.py 오류: {e}")

else:
    st.warning("조건에 맞는 데이터가 없습니다.")
