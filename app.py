import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import RobustScaler

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Crypto Market Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================
# STYLE (MINIMALIS)
# =====================
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: white;
    }
    .metric-label {
        font-size: 14px;
        color: #9aa0a6;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    df = pd.read_csv("crypto_top1000_dataset.csv")
    return df

df = load_data()

# =====================
# CLEANING & FEATURE ENGINEERING
# =====================
df = df.dropna(subset=[
    "high_24h", "low_24h",
    "price_change_24h",
    "price_change_percentage_24h",
    "market_cap_change_24h",
    "market_cap_change_percentage_24h"
])

df["max_supply_available"] = df["max_supply"].notna().astype(int)
df["fdv_available"] = df["fully_diluted_valuation"].notna().astype(int)
df["fdv_mc_ratio"] = df["fully_diluted_valuation"] / df["market_cap"]
df["total_supply"] = df["total_supply"].fillna(df["circulating_supply"])
df["has_1y_history"] = df["price_change_percentage_1y"].notna().astype(int)

df["volatility_24h"] = (df["high_24h"] - df["low_24h"]) / df["current_price"]
df["volume_marketcap_ratio"] = df["total_volume"] / df["market_cap"]
df["supply_inflation_risk"] = 1 - df["supply_utilization"]

# =====================
# OUTLIER CAPPING & SCALING
# =====================
def cap_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return series.clip(q1 - 1.5*iqr, q3 + 1.5*iqr)

scale_cols = [
    "market_cap",
    "total_volume",
    "volatility_24h",
    "volume_marketcap_ratio"
]

for col in scale_cols:
    df[col] = cap_outliers(df[col])

scaler = RobustScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# =====================
# SIDEBAR
# =====================
st.sidebar.title("⚙️ Filter")

rank_range = st.sidebar.slider(
    "Market Cap Rank",
    1, 1000, (1, 100)
)

df_filtered = df[
    (df["market_cap_rank"] >= rank_range[0]) &
    (df["market_cap_rank"] <= rank_range[1])
]

# =====================
# HEADER
# =====================
st.title("📊 Crypto Market Dashboard")
st.markdown(
    "Analisis **market dominance, volatilitas, dan performa harga kripto** secara interaktif."
)

# =====================
# KPI METRICS
# =====================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Koin", len(df_filtered))
col2.metric("Rata-rata Volatilitas", round(df_filtered["volatility_24h"].mean(), 2))
col3.metric("Avg Volume/MarketCap", round(df_filtered["volume_marketcap_ratio"].mean(), 2))
col4.metric("Koin Naik (24h)", int((df_filtered["price_change_percentage_24h"] > 0).sum()))

st.markdown("---")

# =====================
# ROW 1
# =====================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 Top 10 Market Cap")
    top_10 = df.nsmallest(10, "market_cap_rank")
    fig = px.bar(
        top_10,
        x="symbol",
        y="market_cap",
        color="market_cap",
        text_auto=".2s",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📈 Top Gainers vs Losers (24 Jam)")
    top_5 = df_filtered.nlargest(5, "price_change_percentage_24h")
    bot_5 = df_filtered.nsmallest(5, "price_change_percentage_24h")
    combo = pd.concat([top_5, bot_5])

    fig = px.bar(
        combo,
        x="price_change_percentage_24h",
        y="name",
        orientation="h",
        color="price_change_percentage_24h",
        color_continuous_scale="RdYlGn",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================
# ROW 2
# =====================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🗺️ Dominasi Market")
    fig = px.treemap(
        df_filtered,
        path=["symbol"],
        values="market_cap",
        color="price_change_percentage_24h",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🔥 Distribusi Volatilitas")
    fig = px.histogram(
        df_filtered,
        x="price_change_percentage_24h",
        nbins=50,
        template="plotly_dark"
    )
    fig.add_vline(x=0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

# =====================
# ROW 3
# =====================
st.subheader("💰 Harga vs Market Cap")

def categorize(rank):
    if rank <= 10: return "Big Cap"
    if rank <= 50: return "Mid Cap"
    return "Small Cap"

df_filtered["category"] = df_filtered["market_cap_rank"].apply(categorize)

fig = px.scatter(
    df_filtered,
    x="current_price",
    y="market_cap",
    log_x=True,
    log_y=True,
    color="category",
    hover_name="name",
    trendline="ols",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# =====================
# FOOTER
# =====================
st.markdown("---")
st.caption("📌 Data Visualization Project | Streamlit Dashboard")
