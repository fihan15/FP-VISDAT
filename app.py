import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import RobustScaler

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Crypto Market Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# STYLE
# ======================================================
st.markdown("""
<style>
.main { background-color: #0e1117; }
h1,h2,h3,h4 { color: white; }
.stTabs [aria-selected="true"] { background-color: #1e88e5; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    return pd.read_csv("crypto_top1000_dataset.csv")

df = load_data()

# ======================================================
# CLEANING
# ======================================================
df = df.dropna(subset=[
    "high_24h", "low_24h", "current_price",
    "price_change_percentage_24h", "market_cap"
])

# ======================================================
# FEATURE ENGINEERING (RAW)
# ======================================================
df["volatility_24h"] = (df["high_24h"] - df["low_24h"]) / df["current_price"]
df["volume_marketcap_ratio"] = df["total_volume"] / df["market_cap"]

df["total_supply"] = df["total_supply"].fillna(df["circulating_supply"])
df["supply_utilization"] = df["circulating_supply"] / df["total_supply"]
df["supply_utilization"] = df["supply_utilization"].clip(0, 1)

df["supply_inflation_risk"] = 1 - df["supply_utilization"]

# ======================================================
# SAVE RAW VALUES (IMPORTANT)
# ======================================================
df["volatility_raw"] = df["volatility_24h"]
df["volume_mc_raw"] = df["volume_marketcap_ratio"]
df["inflation_raw"] = df["supply_inflation_risk"]

# ======================================================
# SCALING (ONLY FOR VISUAL)
# ======================================================
scale_cols = [
    "market_cap",
    "total_volume",
    "volatility_24h",
    "volume_marketcap_ratio"
]

def cap_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return series.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

for col in scale_cols:
    df[col] = cap_outliers(df[col])

scaler = RobustScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Bubble size
df["size_normalized"] = (
    (df["total_volume"] - df["total_volume"].min()) /
    (df["total_volume"].max() - df["total_volume"].min()) * 30 + 5
)

# ======================================================
# CATEGORY
# ======================================================
def categorize(rank):
    if rank <= 10: return "Big Cap"
    if rank <= 50: return "Mid Cap"
    return "Small Cap"

df["category"] = df["market_cap_rank"].apply(categorize)

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("⚙️ Filter")

rank_range = st.sidebar.slider(
    "Market Cap Rank",
    1, 1000, (1, 100)
)

df_filtered = df[
    (df["market_cap_rank"] >= rank_range[0]) &
    (df["market_cap_rank"] <= rank_range[1])
].copy()

# ======================================================
# HEADER
# ======================================================
st.title("📊 Crypto Market Dashboard")
st.markdown("Analisis **market, performa, dan risiko kripto** secara interaktif")

tabs = st.tabs(["📈 Overview", "📊 Detail", "⚠️ Risk Assessment"])

# ======================================================
# TAB 1: OVERVIEW
# ======================================================
with tabs[0]:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Koin", len(df_filtered))

    with col2:
        st.metric(
            "Rata-rata Volatilitas",
            f"{df_filtered['volatility_raw'].mean():.2%}"
        )

    with col3:
        gain_ratio = (df_filtered["price_change_percentage_24h"] > 0).mean()
        st.metric("Koin Naik (24h)", f"{gain_ratio:.1%}")

    st.subheader("🗺️ Market Dominance")
    fig = px.treemap(
        df_filtered.head(30),
        path=["category", "symbol"],
        values="market_cap",
        color="price_change_percentage_24h",
        color_continuous_scale="RdYlGn",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 2: DETAIL
# ======================================================
with tabs[1]:
    fig = px.scatter(
        df_filtered.head(100),
        x="current_price",
        y="market_cap",
        log_x=True,
        log_y=True,
        color="category",
        size="size_normalized",
        hover_name="name",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 3: RISK ASSESSMENT (FIXED)
# ======================================================
with tabs[2]:
    st.subheader("⚠️ Market Risk Indicator")

    # ===== RISK COMPONENTS (RAW) =====
    volatility_risk = df_filtered["volatility_raw"].mean() * 100
    sentiment_risk = (df_filtered["price_change_percentage_24h"] < 0).mean() * 100
    inflation_risk = df_filtered["inflation_raw"].mean() * 100

    # Clamp
    volatility_risk = np.clip(volatility_risk, 0, 50)
    sentiment_risk = np.clip(sentiment_risk, 0, 100)
    inflation_risk = np.clip(inflation_risk, 0, 100)

    # Composite Score
    risk_score = (
        volatility_risk * 0.4 +
        sentiment_risk * 0.3 +
        inflation_risk * 0.3
    )

    # ===== GAUGE =====
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30], "color": "green"},
                {"range": [30, 70], "color": "yellow"},
                {"range": [70, 100], "color": "red"},
            ],
            "bar": {"color": "darkblue"}
        },
        title={"text": "Tingkat Risiko Pasar"}
    ))

    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)

    # ===== BREAKDOWN =====
    col1, col2, col3 = st.columns(3)
    col1.metric("Volatilitas", f"{volatility_risk:.1f}")
    col2.metric("Sentimen", f"{sentiment_risk:.1f}")
    col3.metric("Inflasi Supply", f"{inflation_risk:.1f}")

    # ===== RADAR CHART =====
    st.subheader("📊 Risiko per Kategori")

    risk_cat = df_filtered.groupby("category").agg({
        "volatility_raw": "mean",
        "inflation_raw": "mean",
        "price_change_percentage_24h": lambda x: (x < 0).mean()
    }).reset_index()

    metrics = ["volatility_raw", "inflation_raw", "price_change_percentage_24h"]

    for col in metrics:
        risk_cat[col] = (
            (risk_cat[col] - risk_cat[col].min()) /
            (risk_cat[col].max() - risk_cat[col].min())
        )

    fig = go.Figure()
    for _, row in risk_cat.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics],
            theta=["Volatility", "Inflation", "Bearish Sentiment"],
            fill="toself",
            name=row["category"]
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1])),
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("📌 Crypto Market Dashboard | Risk logic fixed & stable")
