import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
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
    h1, h2, h3, h4 {
        color: white;
    }
    .metric-label {
        font-size: 14px;
        color: #9aa0a6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e88e5;
    }
    /* Styling untuk insights box */
    .insight-box {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid;
    }
    .success-box {
        background-color: rgba(0, 255, 0, 0.1);
        border-left-color: #00cc00;
    }
    .warning-box {
        background-color: rgba(255, 255, 0, 0.1);
        border-left-color: #ffcc00;
    }
    .danger-box {
        background-color: rgba(255, 0, 0, 0.1);
        border-left-color: #ff3333;
    }
    .info-box {
        background-color: rgba(0, 150, 255, 0.1);
        border-left-color: #1e88e5;
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
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    return series.clip(lower=lower_bound, upper=upper_bound)

scale_cols = [
    "market_cap",
    "total_volume",
    "volatility_24h",
    "volume_marketcap_ratio"
]

# Simpan data asli untuk visualisasi
df["total_volume_original"] = df["total_volume"].copy()

for col in scale_cols:
    df[col] = cap_outliers(df[col])

scaler = RobustScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Normalisasi size untuk visualisasi
df["size_normalized"] = (df["total_volume_original"] - df["total_volume_original"].min()) / \
                        (df["total_volume_original"].max() - df["total_volume_original"].min()) * 30 + 5

# =====================
# HELPER FUNCTIONS
# =====================
def categorize(rank):
    if rank <= 10: return "Big Cap"
    if rank <= 50: return "Mid Cap"
    return "Small Cap"

def get_volatility_label(volatility):
    """Label volatilitas untuk pemula"""
    if volatility < 0.05: return "🟢 Stabil"
    elif volatility < 0.10: return "🟡 Sedang"
    elif volatility < 0.20: return "🟠 Tinggi"
    else: return "🔴 Sangat Tinggi"

def get_sentiment_label(bullish_percent):
    """Label sentimen pasar"""
    if bullish_percent > 60: return "🎯 Optimis"
    elif bullish_percent < 40: return "😰 Pesimis"
    else: return "⚖️ Netral"

def style_price_change(val):
    """Styling untuk perubahan harga"""
    try:
        if isinstance(val, str):
            num_val = float(val.replace('%', '').replace('+', '').replace('$', '').replace(',', ''))
        else:
            num_val = val
        
        if num_val > 5:
            return 'background-color: rgba(0, 255, 0, 0.3); font-weight: bold; color: white'
        elif num_val < -5:
            return 'background-color: rgba(255, 0, 0, 0.3); font-weight: bold; color: white'
        elif num_val > 0:
            return 'background-color: rgba(0, 255, 0, 0.2); color: white'
        else:
            return 'background-color: rgba(255, 0, 0, 0.2); color: white'
    except:
        return ''

# =====================
# SIDEBAR
# =====================
st.sidebar.title("⚙️ Filter & Kontrol")

# Mode untuk pemula
st.sidebar.markdown("---")
st.sidebar.subheader("👶 Mode Pemula")

beginner_mode = st.sidebar.toggle(
    "Aktifkan Mode Sederhana",
    value=True,
    help="Menyederhanakan istilah dan tampilan untuk pemula"
)

# Panduan untuk pemula
with st.sidebar.expander("📖 Panduan Cepat", expanded=False):
    st.markdown("""
    **Glossary:**
    - **Market Cap**: Nilai total pasar = harga × jumlah koin beredar
    - **Volatilitas**: Ukuran fluktuasi harga (semakin tinggi = semakin berisiko)
    - **FDV/MC Ratio**: Perbandingan nilai penuh vs nilai pasar saat ini
    - **Volume/MC Ratio**: Aktivitas trading relatif terhadap ukuran pasar
    - **Top Gainers**: Koin dengan kenaikan harga tertinggi (24 jam)
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filter Data")

# Filter market cap rank
rank_range = st.sidebar.slider(
    "Market Cap Rank" if not beginner_mode else "Peringkat Ukuran Pasar",
    1, 1000, (1, 100),
    help="Filter berdasarkan peringkat market cap (1 = terbesar)"
)

# Filter tambahan untuk pemula
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Filter Performa")

performance_options = ["Semua Koin", "Harga Naik 24h", "Harga Turun 24h", "Volatilitas Tinggi", "Volume Trading Tinggi"]
if beginner_mode:
    performance_options = ["Semua Koin", "Sedang Naik", "Sedang Turun", "Fluktuasi Tinggi", "Trading Aktif"]

performance_filter = st.sidebar.selectbox(
    "Tampilkan koin dengan:",
    performance_options,
    help="Filter berdasarkan performa koin"
)

# Aplikasikan filter
df_filtered = df[
    (df["market_cap_rank"] >= rank_range[0]) &
    (df["market_cap_rank"] <= rank_range[1])
]

# Mapping filter untuk pemula
filter_mapping = {
    "Semua Koin": "Semua Koin",
    "Sedang Naik": "Harga Naik 24h",
    "Sedang Turun": "Harga Turun 24h",
    "Fluktuasi Tinggi": "Volatilitas Tinggi",
    "Trading Aktif": "Volume Trading Tinggi"
}

if beginner_mode:
    actual_filter = filter_mapping.get(performance_filter, performance_filter)
else:
    actual_filter = performance_filter

if actual_filter == "Harga Naik 24h":
    df_filtered = df_filtered[df_filtered["price_change_percentage_24h"] > 0]
elif actual_filter == "Harga Turun 24h":
    df_filtered = df_filtered[df_filtered["price_change_percentage_24h"] < 0]
elif actual_filter == "Volatilitas Tinggi":
    df_filtered = df_filtered[df_filtered["volatility_24h"] > df_filtered["volatility_24h"].quantile(0.75)]
elif actual_filter == "Volume Trading Tinggi":
    df_filtered = df_filtered[df_filtered["volume_marketcap_ratio"] > df_filtered["volume_marketcap_ratio"].quantile(0.75)]

# Kategori market cap
df_filtered["category"] = df_filtered["market_cap_rank"].apply(categorize)

# =====================
# HEADER
# =====================
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📊 Crypto Market Dashboard")
    if beginner_mode:
        st.markdown(
            "**Dashboard sederhana untuk memahami pasar kripto** - Cocok untuk pemula!"
        )
    else:
        st.markdown(
            "Analisis **market dominance, volatilitas, dan performa harga kripto** secara interaktif."
        )
with col2:
    if beginner_mode:
        st.success("👶 **Mode Pemula Aktif**")
    else:
        st.info("ℹ️ Dashboard interaktif untuk analisis pasar kripto")

# =====================
# QUICK INSIGHTS UNTUK PEMULA
# =====================
if beginner_mode and len(df_filtered) > 0:
    st.markdown("---")
    st.subheader("🚀 Insight Cepat untuk Pemula")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        # Insight sentimen pasar
        bullish_percent = (df_filtered["price_change_percentage_24h"] > 0).mean() * 100
        sentiment = get_sentiment_label(bullish_percent)
        sentiment_icon = "📈" if bullish_percent > 60 else "📉" if bullish_percent < 40 else "➡️"
        
        st.markdown(f"""
        <div class="insight-box {'success-box' if bullish_percent > 60 else 'danger-box' if bullish_percent < 40 else 'info-box'}">
            <h4>{sentiment_icon} {sentiment}</h4>
            <p>{bullish_percent:.0f}% koin sedang naik</p>
            <small>{"Bagus untuk beli" if bullish_percent > 60 else "Hati-hati" if bullish_percent < 40 else "Tunggu dulu"}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        # Insight volatilitas
        avg_vol = df_filtered["volatility_24h"].mean()
        vol_label = get_volatility_label(avg_vol)
        vol_icon = "🟢" if avg_vol < 0.05 else "🟡" if avg_vol < 0.10 else "🟠" if avg_vol < 0.20 else "🔴"
        
        st.markdown(f"""
        <div class="insight-box {'success-box' if avg_vol < 0.05 else 'warning-box' if avg_vol < 0.10 else 'danger-box'}">
            <h4>{vol_icon} Risiko Fluktuasi</h4>
            <p>Rata-rata: {avg_vol:.1%}</p>
            <small>{vol_label.split()[1]}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col3:
        # Insight aktivitas trading
        vol_ratio = df_filtered["volume_marketcap_ratio"].median()
        if vol_ratio > 0.1:
            activity = "🔥 Sangat Aktif"
            tip = "Banyak trader"
            icon = "🔥"
            box_class = "success-box"
        elif vol_ratio > 0.05:
            activity = "⚡ Cukup Aktif"
            tip = "Aktivitas normal"
            icon = "⚡"
            box_class = "info-box"
        else:
            activity = "🐌 Kurang Aktif"
            tip = "Trading sepi"
            icon = "🐌"
            box_class = "warning-box"
        
        st.markdown(f"""
        <div class="insight-box {box_class}">
            <h4>{icon} Aktivitas Trading</h4>
            <p>{activity}</p>
            <small>{tip}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Rekomendasi sederhana
    with st.expander("💡 Tips Investasi untuk Pemula", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**💎 Untuk Investor Baru:**")
            st.markdown("""
            1. **Mulai dengan koin besar** (Rank 1-20)
            2. **Cari yang volatilitas rendah** (<5%)
            3. **Volume trading tinggi** (>0.05 ratio)
            4. **Harga sedang naik** (sentimen positif)
            """)
        
        with col2:
            st.markdown("**⚠️ Yang Perlu Dihindari:**")
            st.markdown("""
            1. **Volatilitas ekstrem** (>15%)
            2. **Trading volume rendah**
            3. **Harga terus turun** 7 hari berturut
            4. **Market cap kecil** (Rank >200)
            """)

# =====================
# TABS FOR ORGANIZATION
# =====================
tab_names = ["📈 Overview", "🏆 Top Performers", "📊 Detail Analisis", "⚠️ Risk Assessment"]
if beginner_mode:
    tab_names = ["📊 Ringkasan", "⭐ Top Koin", "🔍 Analisis Detail", "⚠️ Risiko"]

tab1, tab2, tab3, tab4 = st.tabs(tab_names)

# =====================
# TAB 1: OVERVIEW
# =====================
with tab1:
    # KPI METRICS
    st.subheader("📊 Market Snapshot" if not beginner_mode else "📊 Cuplikan Pasar")
    
    with st.expander("📖 Cara Baca Snapshot", expanded=False):
        st.markdown("""
        **Tips untuk Pemula:**
        - **Rata-rata Volatilitas**: 
          🟢 <5% = stabil, 🟡 5-10% = sedang, 🔴 >10% = berisiko
        - **Volume/MarketCap**: 
          <0.05 = aktivitas rendah, >0.1 = sangat aktif
        - **Dominasi Top 10**: 
          >60% = pasar dikuasai koin besar
        - **Rasio Naik/Turun**: 
          >60% = pasar optimis, <40% = pesimis
        """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_coins = len(df_filtered)
        st.metric("Total Koin" if not beginner_mode else "Jumlah Koin", total_coins)
        st.caption("Dalam range yang dipilih")
    
    with col2:
        avg_vol = df_filtered["volatility_24h"].mean()
        vol_label = get_volatility_label(avg_vol).split()[1]
        st.metric(
            "Rata-rata Volatilitas" if not beginner_mode else "Tingkat Fluktuasi", 
            f"{avg_vol:.2%}",
            vol_label
        )
        st.caption("24 jam terakhir")
    
    with col3:
        avg_volume_ratio = df_filtered["volume_marketcap_ratio"].mean()
        ratio_label = "Tinggi" if avg_volume_ratio > 0.1 else "Rendah" if avg_volume_ratio < 0.02 else "Normal"
        st.metric(
            "Avg Volume/MarketCap" if not beginner_mode else "Aktivitas Trading",
            f"{avg_volume_ratio:.3f}",
            ratio_label
        )
        st.caption("Semakin tinggi semakin aktif")
    
    with col4:
        gainers = (df_filtered["price_change_percentage_24h"] > 0).sum()
        total_coins = len(df_filtered)
        gain_percent = (gainers/total_coins*100) if total_coins > 0 else 0
        sentiment = "Naik" if gain_percent > 50 else "Turun"
        st.metric(
            "Koin Naik (24h)" if not beginner_mode else "Naik vs Turun",
            f"{gainers}/{total_coins}", 
            f"{gain_percent:.1f}% ({sentiment})"
        )
    
    # Additional KPIs
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        total_market_cap = df_filtered['market_cap'].sum()
        st.metric(
            "Total Market Cap" if not beginner_mode else "Total Nilai Pasar",
            f"${total_market_cap:,.0f}"
        )
        st.caption("Nilai pasar total")
    
    with col6:
        top_10 = df.nsmallest(10, "market_cap_rank")
        dominance = (top_10['market_cap'].sum() / df_filtered['market_cap'].sum() * 100) if df_filtered['market_cap'].sum() > 0 else 0
        dominance_label = "Terpusat" if dominance > 60 else "Tersebar"
        st.metric(
            "Dominasi Top 10" if not beginner_mode else "Konsentrasi Pasar",
            f"{dominance:.1f}%",
            dominance_label
        )
        st.caption("Semakin tinggi = semakin terpusat")
    
    with col7:
        gain_ratio = (df_filtered['price_change_percentage_24h'] > 0).mean()
        sentiment_label = "Optimis" if gain_ratio > 0.6 else "Pesimis" if gain_ratio < 0.4 else "Netral"
        st.metric(
            "Rasio Naik/Turun" if not beginner_mode else "Sentimen Pasar",
            f"{gain_ratio:.1%}",
            sentiment_label
        )
        st.caption(">60% = optimis, <40% = pesimis")
    
    with col8:
        avg_fdv_ratio = df_filtered['fdv_mc_ratio'].median()
        dilution_label = "Aman" if avg_fdv_ratio < 1.5 else "Hati-hati" if avg_fdv_ratio < 3 else "Risiko"
        st.metric(
            "Avg FDV/MC Ratio" if not beginner_mode else "Potensi Cair",
            f"{avg_fdv_ratio:.2f}",
            dilution_label
        )
        st.caption("<1.5 = aman, >3 = risiko tinggi")
    
    st.markdown("---")
    
    # ROW 1: Market Dominance & Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗺️ Dominasi Market" if not beginner_mode else "🗺️ Peta Pasar")
        
        with st.expander("❓ Cara baca visual ini", expanded=False):
            st.markdown("""
            **Bubble Chart:**
            - **Posisi Horizontal**: Semakin kiri = semakin besar pasar
            - **Posisi Vertikal**: Semakin atas = semakin naik harganya
            - **Ukuran Bubble**: Semakin besar = semakin tinggi nilai pasar
            - **Warna Bubble**: Hijau = naik, Merah = turun
            
            **Tips:**
            - Fokus pada bubble besar di kiri (koin utama)
            - Bubble hijau di atas = peluang bagus
            - Banyak bubble merah = hati-hati
            """)
        
        # Gunakan bubble chart yang lebih mudah dibaca
        display_data = df_filtered.head(30).copy()
        
        fig = px.scatter(
            display_data,
            x="market_cap_rank",
            y="price_change_percentage_24h",
            size="market_cap",
            color="price_change_percentage_24h",
            hover_name="name",
            hover_data={
                "market_cap": ":$.2s",
                "current_price": "$:.2f",
                "price_change_percentage_24h": ":.2f%",
                "volatility_24h": ":.2f",
                "category": True
            },
            color_continuous_scale="RdYlGn",
            size_max=50,
            template="plotly_dark",
            labels={
                "market_cap_rank": "Peringkat (1 = Terbesar)" if not beginner_mode else "Peringkat Ukuran",
                "price_change_percentage_24h": "Perubahan Harga 24h (%)",
                "market_cap": "Market Cap"
            }
        )
        
        # Tambah garis horizontal di 0%
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        
        # Tambah zona kategori
        fig.add_vrect(x0=0, x1=10, fillcolor="green", opacity=0.1, layer="below", line_width=0)
        fig.add_vrect(x0=10, x1=50, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("👉 **Kiri**: Koin besar | **Kanan**: Koin kecil | **Atas**: Naik | **Bawah**: Turun")
    
    with col2:
        st.subheader("📊 Distribusi Perubahan Harga" if not beginner_mode else "📊 Sebaran Naik-Turun")
        
        with st.expander("❓ Apa artinya ini?", expanded=False):
            st.markdown("""
            **Histogram Perubahan Harga:**
            - **Tinggi bar**: Berapa banyak koin dengan perubahan harga tertentu
            - **Posisi bar**: Persentase perubahan harga
            
            **Interpretasi:**
            - Puncak di kanan (hijau) = banyak koin naik
            - Puncak di kiri (merah) = banyak koin turun
            - Sebar merata = pasar bimbang
            """)
        
        fig = px.histogram(
            df_filtered,
            x="price_change_percentage_24h",
            nbins=20,
            template="plotly_dark",
            color_discrete_sequence=['#1e88e5'],
            opacity=0.7
        )
        
        # Highlight area positif dan negatif
        fig.add_vrect(x0=0, x1=df_filtered["price_change_percentage_24h"].max(), 
                     fillcolor="green", opacity=0.1, layer="below", line_width=0)
        fig.add_vrect(x0=df_filtered["price_change_percentage_24h"].min(), x1=0,
                     fillcolor="red", opacity=0.1, layer="below", line_width=0)
        
        fig.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="Netral")
        
        fig.update_layout(
            xaxis_title="Perubahan Harga (%) - 24h",
            yaxis_title="Jumlah Koin"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Hijau = Naik | Merah = Turun | Lebar = Variasi harga")
    
    # ROW 2: Market Health Indicators
    st.subheader("❤️ Market Health Indicators" if not beginner_mode else "❤️ Kesehatan Pasar")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Market Sentiment Gauge
        sentiment_score = df_filtered['price_change_percentage_24h'].mean() * 100 if len(df_filtered) > 0 else 0
        sentiment_label = "Optimis" if sentiment_score > 5 else "Pesimis" if sentiment_score < -5 else "Netral"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentimen Pasar", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [-20, 20], 'tickwidth': 1},
                'bar': {'color': "#1e88e5"},
                'steps': [
                    {'range': [-20, -5], 'color': "red"},
                    {'range': [-5, 5], 'color': "yellow"},
                    {'range': [5, 20], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': sentiment_score
                }
            }
        ))
        
        fig.update_layout(
            template="plotly_dark", 
            height=250,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"📊 {sentiment_label} ({sentiment_score:+.1f}%)")
    
    with col2:
        # Volume Health
        volume_health = df_filtered['volume_marketcap_ratio'].mean() * 100 if len(df_filtered) > 0 else 0
        volume_health = min(max(volume_health, 0), 10)
        health_label = "Aktif" if volume_health > 7 else "Normal" if volume_health > 3 else "Sepi"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=volume_health,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Kesehatan Volume", 'font': {'size': 16}},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 10]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 3], 'color': "red"},
                    {'range': [3, 7], 'color': "yellow"},
                    {'range': [7, 10], 'color': "green"}
                ]
            }
        ))
        
        fig.update_layout(
            template="plotly_dark", 
            height=250,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"📈 {health_label} ({volume_health:.1f}%)")
    
    with col3:
        # Market Cap Distribution
        if len(df_filtered) > 0:
            risk_counts = df_filtered['category'].value_counts()
            
            # Map warna untuk kategori
            color_map = {
                "Big Cap": "#00cc00",
                "Mid Cap": "#ffcc00", 
                "Small Cap": "#ff6666"
            }
            
            colors = [color_map.get(cat, "#1e88e5") for cat in risk_counts.index]
            
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                hole=0.5,
                color=risk_counts.index,
                color_discrete_map=color_map,
                template="plotly_dark"
            )
            
            fig.update_layout(
                title="Distribusi Kapitalisasi",
                showlegend=True,
                height=250,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💎 Besar | ⚖️ Menengah | ⚡ Kecil")
        else:
            st.info("Tidak ada data yang sesuai dengan filter")
    
    # REKOMENDASI UNTUK PEMULA
    if beginner_mode and len(df_filtered) > 0:
        st.markdown("---")
        st.subheader("🎯 Rekomendasi untuk Pemula")
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            # Stable coins untuk pemula
            stable_coins = df_filtered[
                (df_filtered["volatility_24h"] < df_filtered["volatility_24h"].quantile(0.25)) &
                (df_filtered["market_cap_rank"] <= 50) &
                (df_filtered["price_change_percentage_24h"] > 0)
            ].head(5)
            
            if len(stable_coins) > 0:
                st.markdown("**💎 Koin Stabil (Risiko Rendah):**")
                for _, coin in stable_coins.iterrows():
                    volatility_label = get_volatility_label(coin["volatility_24h"])[0]
                    st.markdown(f"""
                    <div style="padding: 8px; margin: 5px 0; background: rgba(0, 255, 0, 0.1); border-radius: 5px;">
                        <strong>{coin['symbol']}</strong> • +{coin['price_change_percentage_24h']:.1f}%<br>
                        <small>{volatility_label} {coin['volatility_24h']:.1%} fluktuasi</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        with rec_col2:
            # Trending coins
            trending_coins = df_filtered[
                (df_filtered["volume_marketcap_ratio"] > df_filtered["volume_marketcap_ratio"].quantile(0.75)) &
                (df_filtered["price_change_percentage_24h"] > 5) &
                (df_filtered["market_cap_rank"] <= 100)
            ].head(5)
            
            if len(trending_coins) > 0:
                st.markdown("**🚀 Sedang Tren (Volume Tinggi):**")
                for _, coin in trending_coins.iterrows():
                    st.markdown(f"""
                    <div style="padding: 8px; margin: 5px 0; background: rgba(255, 200, 0, 0.1); border-radius: 5px;">
                        <strong>{coin['symbol']}</strong> • +{coin['price_change_percentage_24h']:.1f}%<br>
                        <small>Volume: {coin['volume_marketcap_ratio']:.3f}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        with rec_col3:
            # Warning coins
            warning_coins = df_filtered[
                (df_filtered["volatility_24h"] > df_filtered["volatility_24h"].quantile(0.9)) |
                (df_filtered["price_change_percentage_24h"] < -10)
            ].head(5)
            
            if len(warning_coins) > 0:
                st.markdown("**⚠️ Hati-hati (Risiko Tinggi):**")
                for _, coin in warning_coins.iterrows():
                    st.markdown(f"""
                    <div style="padding: 8px; margin: 5px 0; background: rgba(255, 0, 0, 0.1); border-radius: 5px;">
                        <strong>{coin['symbol']}</strong> • {coin['price_change_percentage_24h']:+.1f}%<br>
                        <small>{coin['volatility_24h']:.1%} fluktuasi</small>
                    </div>
                    """, unsafe_allow_html=True)

# =====================
# TAB 2: TOP PERFORMERS
# =====================
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Market Cap" if not beginner_mode else "🏆 10 Koin Terbesar")
        
        with st.expander("💡 Mengapa penting?", expanded=False):
            st.markdown("""
            **Koin besar biasanya:**
            - Lebih stabil (risiko rendah)
            - Likuiditas tinggi (mudah jual/beli)
            - Dipercaya komunitas
            - Cocok untuk pemula
            
            **Contoh:** Bitcoin, Ethereum, BNB
            """)
        
        top_10 = df.nsmallest(10, "market_cap_rank")
        
        # Format untuk display
        display_df = top_10[['symbol', 'name', 'market_cap', 'current_price', 
                           'price_change_percentage_24h']].copy()
        display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"${x:,.0f}")
        display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
        display_df['price_change_percentage_24h'] = display_df['price_change_percentage_24h'].apply(
            lambda x: f"{'+' if x > 0 else ''}{x:.2f}%")
        
        # Apply styling
        def color_price(val):
            if isinstance(val, str) and '+' in val:
                return 'color: green; font-weight: bold'
            elif isinstance(val, str) and '-' in val:
                return 'color: red; font-weight: bold'
            return ''
        
        styled_df = display_df.style.applymap(color_price, subset=['price_change_percentage_24h'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Visual bar chart sederhana
        fig = px.bar(
            top_10,
            x="symbol",
            y="market_cap",
            color="price_change_percentage_24h",
            text="price_change_percentage_24h",
            color_continuous_scale="RdYlGn",
            template="plotly_dark",
            labels={
                "symbol": "Simbol Koin",
                "market_cap": "Market Cap",
                "price_change_percentage_24h": "Perubahan 24h (%)"
            }
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            yaxis_title="Market Cap (Log)",
            yaxis_type="log",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📊 Tinggi bar = Market Cap | Warna = Perubahan harga")
    
    with col2:
        st.subheader("📈 Top Gainers vs Losers" if not beginner_mode else "📈 Terbaik vs Terburuk")
        
        with st.expander("🎯 Strategi trading", expanded=False):
            st.markdown("""
            **Untuk Gainers:**
            - Bisa lanjut naik (momentum)
            - Tapi hati-hati FOMO (beli mahal)
            
            **Untuk Losers:**
            - Bisa diskon (beli murah)
            - Tapi risiko lanjut turun
            
            **Tips:** Jangan langsung ikut tren, analisis dulu!
            """)
        
        if len(df_filtered) > 0:
            # Top gainers
            top_5 = df_filtered.nlargest(5, "price_change_percentage_24h")
            bot_5 = df_filtered.nsmallest(5, "price_change_percentage_24h")
            combo = pd.concat([top_5, bot_5])
            
            # Tambah label
            combo['status'] = ['Gainer']*5 + ['Loser']*5
            
            # Display table
            gainers_df = combo[['symbol', 'name', 'price_change_percentage_24h', 
                              'current_price', 'market_cap', 'status']].copy()
            gainers_df['price_change_percentage_24h'] = gainers_df['price_change_percentage_24h'].apply(
                lambda x: f"{'+' if x > 0 else ''}{x:.2f}%")
            gainers_df['current_price'] = gainers_df['current_price'].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(gainers_df, use_container_width=True, hide_index=True)
            
            # Visual chart horizontal
            fig = px.bar(
                combo,
                x="price_change_percentage_24h",
                y="symbol",
                orientation="h",
                color="status",
                color_discrete_map={"Gainer": "green", "Loser": "red"},
                template="plotly_dark",
                labels={
                    "price_change_percentage_24h": "Perubahan Harga (%)",
                    "symbol": "Simbol Koin",
                    "status": "Status"
                }
            )
            
            fig.update_layout(
                showlegend=True,
                legend_title="Kategori"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🟢 Gainers = Naik | 🔴 Losers = Turun")
        else:
            st.info("Tidak ada data yang sesuai dengan filter")
    
    # Heatmap Performance Sederhana
    st.subheader("🎨 Heatmap Performa - Top 20" if not beginner_mode else "🎨 Perbandingan Performa")
    
    if len(df_filtered) > 0:
        # Pilih 20 koin terbesar dalam filter
        top_20 = df_filtered.nsmallest(20, "market_cap_rank")
        
        # Metrik sederhana untuk pemula
        if beginner_mode:
            performance_metrics = ['price_change_percentage_24h', 'volatility_24h', 'volume_marketcap_ratio']
        else:
            performance_metrics = ['price_change_percentage_24h', 'price_change_percentage_7d',
                                  'volatility_24h', 'volume_marketcap_ratio', 'fdv_mc_ratio']
        
        # Pastikan kolom ada dan tidak NaN
        available_metrics = [m for m in performance_metrics if m in top_20.columns and not top_20[m].isna().all()]
        
        if available_metrics:
            performance_df = top_20.set_index('symbol')[available_metrics]
            
            # Rename columns untuk lebih user-friendly
            metric_names = {
                'price_change_percentage_24h': '24h Return',
                'price_change_percentage_7d': '7d Return',
                'volatility_24h': 'Volatility',
                'volume_marketcap_ratio': 'Volume/MC',
                'fdv_mc_ratio': 'FDV/MC'
            }
            
            if beginner_mode:
                metric_names = {
                    'price_change_percentage_24h': 'Naik/Turun',
                    'volatility_24h': 'Fluktuasi',
                    'volume_marketcap_ratio': 'Aktivitas'
                }
            
            performance_df.columns = [metric_names.get(col, col) for col in available_metrics]
            
            fig = px.imshow(performance_df.T,
                            color_continuous_scale="RdYlGn",
                            aspect="auto",
                            template="plotly_dark",
                            labels=dict(x="Kripto", y="Metrik", color="Nilai"))
            
            fig.update_layout(
                height=400,
                title="Semakin hijau = semakin baik"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Legenda sederhana
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("🟢 **Hijau** = Baik")
            with col2:
                st.markdown("🟡 **Kuning** = Netral")
            with col3:
                st.markdown("🔴 **Merah** = Buruk")
        else:
            st.info("Metrik performa tidak tersedia")
    else:
        st.info("Tidak ada data yang sesuai dengan filter")

# =====================
# TAB 3: DETAIL ANALISIS
# =====================
with tab3:
    st.subheader("💰 Analisis Harga vs Market Cap" if not beginner_mode else "💰 Hubungan Harga & Ukuran")
    
    with st.expander("📈 Cara membaca scatter plot", expanded=False):
        st.markdown("""
        **Interpretasi:**
        - **Atas kanan**: Harga tinggi & pasar besar = established
        - **Atas kiri**: Harga tinggi & pasar kecil = overvalued?
        - **Bawah kanan**: Harga rendah & pasar besar = undervalued?
        - **Bawah kiri**: Harga rendah & pasar kecil = risky
        
        **Tips:** Cari yang di kanan bawah (harga murah, pasar besar)
        """)
    
    if len(df_filtered) > 0:
        # Batasi data untuk visual yang lebih jelas
        scatter_data = df_filtered.head(50).copy()
        
        # Tambah kategori harga
        scatter_data['price_category'] = pd.qcut(scatter_data['current_price'], 
                                                q=3, 
                                                labels=['Murah', 'Sedang', 'Mahal'])
        
        fig = px.scatter(
            scatter_data,
            x="current_price",
            y="market_cap",
            color="category",
            hover_name="name",
            hover_data={
                "price_change_percentage_24h": ":.1f%",
                "volatility_24h": ":.2f",
                "volume_marketcap_ratio": ":.3f",
                "market_cap_rank": True,
                "category": False
            },
            size="size_normalized",
            template="plotly_dark",
            labels={
                "current_price": "Harga Saat Ini (USD)",
                "market_cap": "Market Cap",
                "category": "Kategori Pasar"
            },
            log_x=True,
            log_y=True
        )
        
        # Tambah quadrant lines
        median_price = scatter_data['current_price'].median()
        median_mcap = scatter_data['market_cap'].median()
        
        fig.add_hline(y=median_mcap, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=median_price, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Tambah anotasi quadrant
        fig.add_annotation(x=median_price*10, y=median_mcap*10, text="Harga & Pasar Besar", 
                          showarrow=False, font=dict(color="white", size=10))
        fig.add_annotation(x=median_price/10, y=median_mcap*10, text="Murah, Pasar Besar", 
                          showarrow=False, font=dict(color="white", size=10))
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Ukuran bubble: Volume Trading | Warna: Kategori Market Cap")
    else:
        st.info("Tidak ada data yang sesuai dengan filter")
    
    # Correlation Matrix Sederhana
    st.subheader("🔗 Hubungan antar Variabel" if not beginner_mode else "🔗 Keterkaitan")
    
    with st.expander("🤔 Apa artinya korelasi?", expanded=False):
        st.markdown("""
        **Korelasi:**
        - **+1.0**: Sempurna searah (naik bersama)
        - **0.0**: Tidak ada hubungan
        - **-1.0**: Sempurna berlawanan (satu naik, satu turun)
        
        **Contoh:**
        - Harga & Market Cap biasanya + (searah)
        - Volatilitas & Volume bisa + (aktif = fluktuatif)
        """)
    
    if len(df_filtered) > 0:
        if beginner_mode:
            numeric_cols = ['current_price', 'market_cap', 'total_volume', 
                           'price_change_percentage_24h', 'volatility_24h']
        else:
            numeric_cols = ['current_price', 'market_cap', 'total_volume', 
                           'price_change_percentage_24h', 'volatility_24h',
                           'volume_marketcap_ratio', 'fdv_mc_ratio']
        
        # Hanya ambil kolom yang ada
        available_numeric = [col for col in numeric_cols if col in df_filtered.columns]
        
        if len(available_numeric) > 1:
            corr_df = df_filtered[available_numeric].corr()
            
            # Rename columns untuk pemula
            if beginner_mode:
                rename_dict = {
                    'current_price': 'Harga',
                    'market_cap': 'Market Cap',
                    'total_volume': 'Volume',
                    'price_change_percentage_24h': 'Perubahan',
                    'volatility_24h': 'Fluktuasi'
                }
                corr_df = corr_df.rename(columns=rename_dict, index=rename_dict)
            
            fig = px.imshow(corr_df,
                            color_continuous_scale="RdBu",
                            zmin=-1, zmax=1,
                            text_auto=".2f",
                            template="plotly_dark",
                            title="Semakin biru = searah, Semakin merah = berlawanan")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak cukup data untuk analisis korelasi")
    else:
        st.info("Tidak ada data yang sesuai dengan filter")
    
    # Summary Table dengan Filter
    st.subheader("📋 Daftar Koin" if not beginner_mode else "📋 Ringkasan Data")
    
    if len(df_filtered) > 0:
        # Filter kolom untuk pemula
        if beginner_mode:
            summary_cols = ['name', 'symbol', 'current_price', 'price_change_percentage_24h',
                           'market_cap', 'volatility_24h', 'category']
        else:
            summary_cols = ['name', 'symbol', 'current_price', 'price_change_percentage_24h',
                           'market_cap', 'volatility_24h', 'volume_marketcap_ratio', 'category']
        
        # Hanya ambil kolom yang ada
        available_summary = [col for col in summary_cols if col in df_filtered.columns]
        
        # Limit jumlah baris
        num_rows = st.slider("Jumlah koin ditampilkan:", 10, 50, 20)
        summary_df = df_filtered.head(num_rows)[available_summary].copy()
        
        # Formatting
        if 'current_price' in summary_df.columns:
            summary_df['current_price'] = summary_df['current_price'].apply(lambda x: f"${x:,.2f}")
        
        if 'price_change_percentage_24h' in summary_df.columns:
            summary_df['price_change_percentage_24h'] = summary_df['price_change_percentage_24h'].apply(
                lambda x: f"{'+' if x > 0 else ''}{x:.2f}%")
        
        if 'market_cap' in summary_df.columns:
            summary_df['market_cap'] = summary_df['market_cap'].apply(lambda x: f"${x:,.0f}")
        
        if 'volatility_24h' in summary_df.columns:
            summary_df['volatility_24h'] = summary_df['volatility_24h'].apply(lambda x: f"{x:.2%}")
        
        if 'volume_marketcap_ratio' in summary_df.columns:
            summary_df['volume_marketcap_ratio'] = summary_df['volume_marketcap_ratio'].apply(lambda x: f"{x:.3f}")
        
        # Styling
        styled_df = summary_df.style.map(style_price_change, subset=['price_change_percentage_24h'])
        
        st.dataframe(styled_df, use_container_width=True, height=400, hide_index=True)
        
        # Download option
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download data sebagai CSV",
            data=csv,
            file_name="crypto_data.csv",
            mime="text/csv"
        )
    else:
        st.info("Tidak ada data yang sesuai dengan filter")

# =====================
# TAB 4: RISK ASSESSMENT
# =====================
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Indikator Risiko Pasar" if not beginner_mode else "⚠️ Tingkat Risiko")
        
        with st.expander("📊 Komponen risiko", expanded=False):
            st.markdown("""
            **Risiko dihitung dari:**
            1. **Volatilitas (40%)**: Seberapa besar harga naik-turun
            2. **Sentimen (30%)**: Berapa % koin sedang turun
            3. **Inflasi (30%)**: Potensi tambahan supply
            
            **Skor Risiko:**
            - 🟢 0-30: Risiko rendah (aman)
            - 🟡 30-70: Risiko sedang (hati-hati)
            - 🔴 70-100: Risiko tinggi (sangat berhati-hati)
            """)
        
        # Hitung risk score komposit
        if len(df_filtered) > 0:
            volatility_risk = df_filtered['volatility_24h'].mean() * 100
            sentiment_risk = (df_filtered['price_change_percentage_24h'] < 0).mean() * 100
            inflation_risk = df_filtered['supply_inflation_risk'].mean() * 100 if 'supply_inflation_risk' in df_filtered.columns else 0
            
            risk_score = (volatility_risk * 0.4 + sentiment_risk * 0.3 + inflation_risk * 0.3)
            risk_score = min(max(risk_score, 0), 100)
            
            # Risk level
            if risk_score < 30:
                risk_level = "🟢 Rendah"
                risk_color = "green"
            elif risk_score < 70:
                risk_level = "🟡 Sedang"
                risk_color = "yellow"
            else:
                risk_level = "🔴 Tinggi"
                risk_color = "red"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Tingkat Risiko: {risk_level}", 'font': {'size': 18}},
                delta={'reference': 50},
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_score
                    }
                }
            ))
            
            fig.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk breakdown dengan progress bars
            st.markdown("**Detail Risiko:**")
            
            def progress_bar(value, label, color):
                progress_html = f"""
                <div style="margin: 5px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{label}</span>
                        <span>{value:.1f}%</span>
                    </div>
                    <div style="height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden;">
                        <div style="height: 100%; width: {value}%; background: {color};"></div>
                    </div>
                </div>
                """
                return progress_html
            
            st.markdown(progress_bar(volatility_risk, "Volatilitas", "#ff6b6b"), unsafe_allow_html=True)
            st.markdown(progress_bar(sentiment_risk, "Sentimen Negatif", "#feca57"), unsafe_allow_html=True)
            st.markdown(progress_bar(inflation_risk, "Risiko Inflasi", "#1dd1a1"), unsafe_allow_html=True)
        else:
            st.info("Tidak ada data untuk kalkulasi risiko")
    
    with col2:
        st.subheader("📊 Perbandingan Risiko per Kategori" if not beginner_mode else "📊 Risiko per Ukuran")
        
        if len(df_filtered) > 0 and 'category' in df_filtered.columns:
            # Hitung risk metrics per kategori
            risk_summary = df_filtered.groupby('category').agg({
                'name': 'count',
                'volatility_24h': 'mean',
                'price_change_percentage_24h': 'mean'
            }).rename(columns={'name': 'jumlah_koin'}).reset_index()
            
            # Tampilkan tabel sederhana
            display_risk = risk_summary.copy()
            display_risk['volatility_24h'] = display_risk['volatility_24h'].apply(lambda x: f"{x:.2%}")
            display_risk['price_change_percentage_24h'] = display_risk['price_change_percentage_24h'].apply(
                lambda x: f"{'+' if x > 0 else ''}{x:.2f}%")
            
            st.dataframe(display_risk, use_container_width=True, hide_index=True)
            
            # Buat bar chart sederhana
            fig = px.bar(
                risk_summary,
                x='category',
                y=['volatility_24h'],
                barmode='group',
                template='plotly_dark',
                color_discrete_sequence=['#ff6b6b'],
                labels={
                    'value': 'Volatilitas',
                    'category': 'Kategori',
                    'variable': 'Metrik'
                }
            )
            
            fig.update_layout(
                title="Rata-rata Volatilitas per Kategori",
                yaxis_tickformat=".0%",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Semakin tinggi = semakin berisiko")
        else:
            st.info("Data kategori tidak tersedia")
    
    # High Risk Coins dengan Penjelasan
    st.subheader("🔴 Koin dengan Risiko Tinggi" if not beginner_mode else "🔴 Perhatian!")
    
    with st.expander("🤔 Mengapa berisiko?", expanded=False):
        st.markdown("""
        **Ciri koin berisiko tinggi:**
        1. **Fluktuasi ekstrem** (>15% per hari)
        2. **Trend turun terus** (7 hari negatif)
        3. **Volume rendah** (sulit jual)
        4. **Market cap kecil** (mudid dimanipulasi)
        
        **Untuk pemula:** Hindari atau alokasi kecil saja!
        """)
    
    if len(df_filtered) > 0:
        # Hitung risk score untuk setiap koin
        df_filtered['risk_score_temp'] = (
            df_filtered['volatility_24h'].rank(pct=True) * 0.4 +
            (df_filtered['price_change_percentage_24h'] < 0).astype(int) * 0.3 +
            (df_filtered['price_change_percentage_24h'] < -5).astype(int) * 0.3
        )
        
        # Tambah warning flag
        df_filtered['warning'] = df_filtered.apply(lambda row: 
            "⚠️" if row['volatility_24h'] > df_filtered['volatility_24h'].quantile(0.9) else
            "🔴" if row['price_change_percentage_24h'] < -10 else
            "🟡" if row['price_change_percentage_24h'] < -5 else "🟢", axis=1)
        
        high_risk = df_filtered.nlargest(10, 'risk_score_temp')[['warning', 'symbol', 'name', 'risk_score_temp', 
                                                               'volatility_24h', 
                                                               'price_change_percentage_24h',
                                                               'category']].copy()
        
        # Format untuk display
        high_risk['risk_score'] = high_risk['risk_score_temp'].apply(lambda x: f"{x:.1%}")
        high_risk['volatility_24h'] = high_risk['volatility_24h'].apply(lambda x: f"{x:.2%}")
        high_risk['price_change_percentage_24h'] = high_risk['price_change_percentage_24h'].apply(
            lambda x: f"{'+' if x > 0 else ''}{x:.2f}%")
        
        # Reorder columns
        high_risk = high_risk[['warning', 'symbol', 'name', 'price_change_percentage_24h', 
                              'volatility_24h', 'risk_score', 'category']]
        
        st.dataframe(high_risk, use_container_width=True, hide_index=True)
        
        # Tambah insight
        if len(high_risk) > 0:
            high_risk_count = len(high_risk[high_risk['warning'].isin(['⚠️', '🔴'])])
            if high_risk_count > 5:
                st.warning(f"⚠️ **Peringatan:** {high_risk_count} koin sangat berisiko!")
            elif high_risk_count > 0:
                st.info(f"ℹ️ Ada {high_risk_count} koin yang perlu diwaspadai")
    else:
        st.info("Tidak ada data untuk analisis risiko")
    
    # Risk Mitigation Tips untuk Pemula
    if beginner_mode:
        st.markdown("---")
        st.subheader("🛡️ Tips Mengurangi Risiko")
        
        tip_col1, tip_col2 = st.columns(2)
        
        with tip_col1:
            st.markdown("""
            **💼 Strategi Investasi:**
            1. **Diversifikasi**: Jangan semua di 1 koin
            2. **DCA**: Beli bertahap, bukan sekaligus
            3. **Stop Loss**: Batas maksimal rugi
            4. **Take Profit**: Ambil untung berkala
            """)
        
        with tip_col2:
            st.markdown("""
            **📚 Prinsip Utama:**
            1. **Hanya invest uang dingin**
            2. **Riset sebelum beli**
            3. **Jangan ikut FOMO**
            4. **Portfolio balance**: 70% besar, 30% kecil
            """)

# =====================
# FOOTER
# =====================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    if beginner_mode:
        st.caption("📌 Dashboard Crypto untuk Pemula")
    else:
        st.caption("📌 Data Visualization Project | Streamlit Dashboard")

with footer_col2:
    st.caption(f"🔄 Data diperbarui: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

with footer_col3:
    st.caption(f"🔍 Total koin ditampilkan: {len(df_filtered)}")
