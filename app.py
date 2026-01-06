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
# STYLE (MINIMALIS & RAMAH PEMULA)
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
    .insight-box {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #1e88e5;
    }
    .risk-box {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .tip-box {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #444;
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

def get_sentiment_emoji(score):
    if score > 5:
        return "😊", "Positif", "green"
    elif score < -5:
        return "😟", "Negatif", "red"
    else:
        return "😐", "Netral", "yellow"

def get_risk_level(score):
    if score <= 30:
        return "🟢 Rendah", "Pasar stabil, risiko minimal", "#00cc00"
    elif score <= 70:
        return "🟡 Sedang", "Pasar cukup volatil, hati-hati", "#ffcc00"
    else:
        return "🔴 Tinggi", "Pasar sangat volatil, risiko tinggi", "#ff3333"

# =====================
# SIDEBAR
# =====================
st.sidebar.title("⚙️ Filter & Kontrol")

# Panduan untuk pemula
with st.sidebar.expander("📖 Panduan Cepat untuk Pemula", expanded=True):
    st.markdown("""
    **Untuk Anda yang baru mulai:**
    
    **🎯 Fokus pada:**
    1. **Market Cap Rank** - Peringkat berdasarkan ukuran pasar
    2. **Perubahan Harga 24h** - Naik/turun dalam sehari
    3. **Volume Trading** - Seberapa aktif diperdagangkan
    
    **💡 Tips:**
    - Mulai dari **koin besar** (Rank 1-20) - lebih stabil
    - Cek **Volume/Market Cap Ratio** > 0.05 = likuiditas baik
    - Hindari **volatilitas tinggi** jika pemula
    
    **📊 Istilah Penting:**
    - **Market Cap**: Nilai total pasar
    - **Volatilitas**: Tingkat fluktuasi harga
    - **FDV/MC**: Potensi penambahan supply
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filter Data")

# Filter market cap rank
rank_range = st.sidebar.slider(
    "Peringkat Market Cap",
    1, 1000, (1, 100),
    help="1 = terbesar (Bitcoin), 100 = lebih kecil"
)

# Filter tambahan untuk pemula
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Filter Performa")

performance_filter = st.sidebar.selectbox(
    "Tampilkan koin dengan:",
    ["Semua Koin", "Harga Naik 24h", "Harga Turun 24h", "Volatilitas Rendah", "Volume Trading Tinggi"],
    help="Volatilitas Rendah = lebih stabil untuk pemula"
)

# Aplikasikan filter
df_filtered = df[
    (df["market_cap_rank"] >= rank_range[0]) &
    (df["market_cap_rank"] <= rank_range[1])
]

if performance_filter == "Harga Naik 24h":
    df_filtered = df_filtered[df_filtered["price_change_percentage_24h"] > 0]
elif performance_filter == "Harga Turun 24h":
    df_filtered = df_filtered[df_filtered["price_change_percentage_24h"] < 0]
elif performance_filter == "Volatilitas Rendah":
    df_filtered = df_filtered[df_filtered["volatility_24h"] < df_filtered["volatility_24h"].quantile(0.25)]
elif performance_filter == "Volume Trading Tinggi":
    df_filtered = df_filtered[df_filtered["volume_marketcap_ratio"] > df_filtered["volume_marketcap_ratio"].quantile(0.75)]

# Kategori market cap
df_filtered["category"] = df_filtered["market_cap_rank"].apply(categorize)

# =====================
# HEADER
# =====================
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📊 Dashboard Pasar Crypto")
    st.markdown(
        "**Dashboard interaktif untuk memahami pasar crypto dengan mudah**"
    )
with col2:
    st.info("🎯 **Tips**: Gunakan filter di sidebar untuk mulai eksplorasi")

# =====================
# TABS FOR ORGANIZATION
# =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Overview", "🏆 Top Performers", "📊 Detail Analisis", "⚠️ Risk Assessment", "🚀 Quick Insights"])

# =====================
# TAB 1: OVERVIEW - DIPERBAIKI
# =====================
with tab1:
    # KPI METRICS - Sederhana untuk pemula
    st.subheader("📊 Snapshoot Pasar Hari Ini")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_market_cap = df_filtered['market_cap'].sum()
        st.metric("💰 Total Nilai Pasar", f"${total_market_cap:,.0f}")
        st.caption("Nilai semua koin dalam filter")
    
    with col2:
        avg_change = df_filtered["price_change_percentage_24h"].mean()
        emoji, label, color = get_sentiment_emoji(avg_change)
        st.metric("📈 Sentimen Pasar", f"{avg_change:+.1f}%", label=f"{emoji} {label}")
        st.caption("Rata-rata perubahan harga")
    
    with col3:
        gainers = (df_filtered["price_change_percentage_24h"] > 0).sum()
        total_coins = len(df_filtered)
        ratio = (gainers/total_coins*100) if total_coins > 0 else 0
        st.metric("📊 Koin Naik/Turun", f"{gainers}/{total_coins}", f"{ratio:.1f}%")
        st.caption("Lebih banyak hijau = pasar sehat")
    
    with col4:
        avg_volume_ratio = df_filtered["volume_marketcap_ratio"].mean()
        volume_status = "🟢 Aktif" if avg_volume_ratio > 0.05 else "🟡 Tenang"
        st.metric("💎 Aktivitas Trading", f"{avg_volume_ratio:.3f}", volume_status)
        st.caption("Semakin tinggi = semakin cair")
    
    st.markdown("---")
    
    # INSIGHT BOX UNTUK PEMULA
    st.markdown("""
    <div class="insight-box">
        <h4>🎯 Apa yang Harus Diperhatikan?</h4>
        <ul>
            <li><b>Market Cap Besar</b> = Lebih stabil, risiko lebih rendah</li>
            <li><b>Volume Trading Tinggi</b> = Mudah beli/jual</li>
            <li><b>Perubahan Hijau</b> = Banyak koin naik</li>
            <li><b>Sentimen Positif</b> = Pasar optimis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # ROW 1: Market Dominance & Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗺️ Dominasi Pasar - Top 30")
        st.markdown("💡 **Cara baca**: Ukuran kotak = nilai pasar, Warna = naik/hijau, turun/merah")
        
        top_30 = df_filtered.head(30)
        if len(top_30) > 0:
            fig = px.treemap(
                top_30,
                path=["category", "symbol"],
                values="market_cap",
                color="price_change_percentage_24h",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                template="plotly_dark",
                hover_data=["current_price", "market_cap_rank"]
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Sebaran Perubahan Harga")
        st.markdown("💡 **Cara baca**: Grafik menunjukkan berapa banyak koin naik/turun")
        
        if len(df_filtered) > 0:
            # Buat kategori sederhana
            df_filtered["change_category"] = pd.cut(
                df_filtered["price_change_percentage_24h"],
                bins=[-100, -10, -5, 0, 5, 10, 100],
                labels=["Turun Banyak (>10%)", "Turun Sedang", "Turun Kecil", 
                       "Naik Kecil", "Naik Sedang", "Naik Banyak (>10%)"]
            )
            
            fig = px.bar(
                df_filtered["change_category"].value_counts().sort_index(),
                orientation="v",
                color_discrete_sequence=['#1e88e5'],
                template="plotly_dark",
                labels={"value": "Jumlah Koin", "index": "Kategori Perubahan"}
            )
            fig.add_vline(x=2.5, line_dash="dash", line_color="white", annotation_text="Netral")
            st.plotly_chart(fig, use_container_width=True)
    
    # ROW 2: Market Health Indicators - SEDERHANAKAN
    st.subheader("❤️ Kesehatan Pasar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market Sentiment - Sederhana
        if len(df_filtered) > 0:
            sentiment_score = df_filtered['price_change_percentage_24h'].mean()
            emoji, label, color = get_sentiment_emoji(sentiment_score)
            
            st.markdown(f"""
            <div class="risk-box" style="border-left: 5px solid {color};">
                <h3>{emoji}</h3>
                <h1>{label}</h1>
                <p>Perubahan rata-rata: <b>{sentiment_score:+.1f}%</b></p>
                <p style="font-size: 12px; color: #aaa;">{len(df_filtered)} koin dianalisis</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Market Stability - Sederhana
        if len(df_filtered) > 0:
            volatility_avg = df_filtered['volatility_24h'].mean() * 100
            if volatility_avg < 5:
                stability = "🟢 Stabil"
                stability_desc = "Fluktuasi harga rendah"
                stability_color = "#00cc00"
            elif volatility_avg < 15:
                stability = "🟡 Normal"
                stability_desc = "Fluktuasi harga wajar"
                stability_color = "#ffcc00"
            else:
                stability = "🔴 Volatil"
                stability_desc = "Fluktuasi harga tinggi"
                stability_color = "#ff3333"
            
            st.markdown(f"""
            <div class="risk-box" style="border-left: 5px solid {stability_color};">
                <h3>{stability.split()[0]}</h3>
                <h1>{stability.split()[1]}</h1>
                <p>Volatilitas: <b>{volatility_avg:.1f}%</b></p>
                <p style="font-size: 12px; color: #aaa;">{stability_desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # TIPS BOX
    st.markdown("""
    <div class="tip-box">
        <h4>💡 Tips untuk Hari Ini:</h4>
        <ol>
            <li><b>Market Cap Rank 1-20</b>: Pilihan aman untuk pemula</li>
            <li><b>Cek volume trading</b>: Pastikan > 0.05 untuk likuiditas baik</li>
            <li><b>Diversifikasi</b>: Jangan fokus pada 1 koin saja</li>
            <li><b>Pelajari dulu</b>: Mulai dengan jumlah kecil untuk belajar</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# =====================
# TAB 2: TOP PERFORMERS - DIPERBAIKI
# =====================
with tab2:
    st.subheader("🏆 Pemain Terbaik Hari Ini")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Top 5 Naik Terbanyak")
        
        if len(df_filtered) > 0:
            top_gainers = df_filtered.nlargest(5, "price_change_percentage_24h").copy()
            
            # Format untuk display
            display_gainers = top_gainers[['symbol', 'name', 'price_change_percentage_24h', 
                                         'current_price', 'market_cap_rank']].copy()
            display_gainers['price_change'] = display_gainers['price_change_percentage_24h'].apply(
                lambda x: f"{'+' if x > 0 else ''}{x:.1f}%")
            display_gainers['current_price'] = display_gainers['current_price'].apply(
                lambda x: f"${x:,.2f}")
            display_gainers['Rank'] = display_gainers['market_cap_rank']
            
            display_gainers = display_gainers[['symbol', 'name', 'price_change', 
                                             'current_price', 'Rank']]
            
            # Tampilkan dengan warna
            for _, row in display_gainers.iterrows():
                with st.container():
                    col_a, col_b, col_c = st.columns([1, 2, 2])
                    with col_a:
                        st.markdown(f"**{row['symbol']}**")
                    with col_b:
                        st.markdown(f"<span style='color: green; font-weight: bold;'>{row['price_change']}</span>", 
                                  unsafe_allow_html=True)
                    with col_c:
                        st.markdown(f"{row['current_price']}")
            
            # Bar chart sederhana
            fig = px.bar(
                top_gainers,
                x="symbol",
                y="price_change_percentage_24h",
                color="price_change_percentage_24h",
                color_continuous_scale="greens",
                template="plotly_dark",
                labels={"price_change_percentage_24h": "Kenaikan (%)", "symbol": "Koin"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📉 Top 5 Turun Terbanyak")
        
        if len(df_filtered) > 0:
            top_losers = df_filtered.nsmallest(5, "price_change_percentage_24h").copy()
            
            # Format untuk display
            display_losers = top_losers[['symbol', 'name', 'price_change_percentage_24h', 
                                       'current_price', 'market_cap_rank']].copy()
            display_losers['price_change'] = display_losers['price_change_percentage_24h'].apply(
                lambda x: f"{x:.1f}%")
            display_losers['current_price'] = display_losers['current_price'].apply(
                lambda x: f"${x:,.2f}")
            display_losers['Rank'] = display_losers['market_cap_rank']
            
            display_losers = display_losers[['symbol', 'name', 'price_change', 
                                           'current_price', 'Rank']]
            
            # Tampilkan dengan warna
            for _, row in display_losers.iterrows():
                with st.container():
                    col_a, col_b, col_c = st.columns([1, 2, 2])
                    with col_a:
                        st.markdown(f"**{row['symbol']}**")
                    with col_b:
                        st.markdown(f"<span style='color: red; font-weight: bold;'>{row['price_change']}</span>", 
                                  unsafe_allow_html=True)
                    with col_c:
                        st.markdown(f"{row['current_price']}")
            
            # Bar chart sederhana
            fig = px.bar(
                top_losers,
                x="symbol",
                y="price_change_percentage_24h",
                color="price_change_percentage_24h",
                color_continuous_scale="reds",
                template="plotly_dark",
                labels={"price_change_percentage_24h": "Penurunan (%)", "symbol": "Koin"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # PERBANDINGAN PERFORMANCE
    st.subheader("📊 Perbandingan Performa - Top 10")
    
    if len(df_filtered) > 0:
        top_10 = df_filtered.nsmallest(10, "market_cap_rank").copy()
        
        # Buat chart side-by-side yang mudah dibaca
        metrics = {
            'price_change_percentage_24h': 'Perubahan 24h (%)',
            'volatility_24h': 'Volatilitas (%)'
        }
        
        fig = go.Figure()
        
        # Tambah bar untuk perubahan harga
        fig.add_trace(go.Bar(
            name='Perubahan Harga',
            x=top_10['symbol'],
            y=top_10['price_change_percentage_24h'],
            marker_color=['green' if x > 0 else 'red' for x in top_10['price_change_percentage_24h']],
            yaxis='y'
        ))
        
        # Tambah line untuk volatilitas
        fig.add_trace(go.Scatter(
            name='Volatilitas',
            x=top_10['symbol'],
            y=top_10['volatility_24h'] * 100,
            mode='lines+markers',
            line=dict(color='yellow', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Perubahan Harga vs Volatilitas",
            yaxis=dict(title="Perubahan Harga (%)"),
            yaxis2=dict(title="Volatilitas (%)", overlaying="y", side="right"),
            template="plotly_dark",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("💡 **Biru/Hijau** = harga naik, **Merah** = harga turun, **Garis kuning** = tingkat fluktuasi")

# =====================
# TAB 3: DETAIL ANALISIS - DIPERBAIKI
# =====================
with tab3:
    st.subheader("🔍 Analisis Detail Koin")
    
    # Pilih koin untuk analisis detail
    coin_list = df_filtered['symbol'].tolist()
    if coin_list:
        selected_coin = st.selectbox("🎯 Pilih koin untuk analisis detail:", coin_list[:50])
        
        # Tampilkan detail koin
        coin_data = df_filtered[df_filtered['symbol'] == selected_coin].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px;">
                <h4>💰 Harga</h4>
                <h2>${coin_data['current_price']:,.2f}</h2>
                <p style="color: {'green' if coin_data['price_change_percentage_24h'] > 0 else 'red'};">
                    {('+' if coin_data['price_change_percentage_24h'] > 0 else '')}{coin_data['price_change_percentage_24h']:.2f}% (24h)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px;">
                <h4>🏆 Peringkat</h4>
                <h2>#{coin_data['market_cap_rank']}</h2>
                <p>Kategori: {coin_data['category']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px;">
                <h4>📊 Aktivitas</h4>
                <p>Volume/Market Cap: {coin_data['volume_marketcap_ratio']:.4f}</p>
                <p>Volatilitas: {coin_data['volatility_24h']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # SCATTER PLOT dengan penjelasan
        st.subheader("📈 Posisi di Pasar")
        
        st.markdown("""
        <div class="insight-box">
            <h4>🎯 Cara Membaca Grafik:</h4>
            <ul>
                <li><b>Kanan Atas</b>: Koin besar & mahal (contoh: Bitcoin)</li>
                <li><b>Kiri Bawah</b>: Koin kecil & murah</li>
                <li><b>Ukuran Bulatan</b>: Volume trading (besar = aktif)</li>
                <li><b>Warna</b>: Biru=Big Cap, Hijau=Mid Cap, Merah=Small Cap</li>
                <li><b>Bintang</b>: Koin yang Anda pilih</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if len(df_filtered) > 0:
            scatter_data = df_filtered.head(100).copy()
            
            # Tandai koin yang dipilih
            scatter_data['selected'] = scatter_data['symbol'] == selected_coin
            
            fig = px.scatter(
                scatter_data,
                x="current_price",
                y="market_cap",
                log_x=True,
                log_y=True,
                color="category",
                hover_name="name",
                hover_data={
                    "price_change_percentage_24h": ":.2f%",
                    "volatility_24h": ":.2%",
                    "volume_marketcap_ratio": ":.4f",
                    "market_cap_rank": True,
                    "category": False,
                    "selected": False
                },
                size="size_normalized",
                template="plotly_dark",
                labels={
                    "current_price": "Harga (USD)",
                    "market_cap": "Market Cap",
                    "category": "Kategori"
                },
                symbol="selected",
                symbol_map={True: "star", False: "circle"},
                size_max=20
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("💡 **Tips**: Arahkan mouse ke bulatan untuk info detail")
    
    # COMPARISON TOOL
    st.subheader("⚖️ Bandingkan 2 Koin")
    
    col1, col2 = st.columns(2)
    
    with col1:
        coin1 = st.selectbox("Pilih koin pertama:", coin_list[:30] if coin_list else [], key="coin1")
    
    with col2:
        coin2 = st.selectbox("Pilih koin kedua:", coin_list[:30] if coin_list else [], key="coin2")
    
    if coin1 and coin2 and coin1 != coin2:
        coin1_data = df_filtered[df_filtered['symbol'] == coin1].iloc[0]
        coin2_data = df_filtered[df_filtered['symbol'] == coin2].iloc[0]
        
        comparison_df = pd.DataFrame({
            'Metrik': ['Harga', 'Perubahan 24h', 'Peringkat', 'Volatilitas', 'Volume/MC Ratio'],
            coin1: [
                f"${coin1_data['current_price']:,.2f}",
                f"{coin1_data['price_change_percentage_24h']:+.2f}%",
                f"#{coin1_data['market_cap_rank']}",
                f"{coin1_data['volatility_24h']:.2%}",
                f"{coin1_data['volume_marketcap_ratio']:.4f}"
            ],
            coin2: [
                f"${coin2_data['current_price']:,.2f}",
                f"{coin2_data['price_change_percentage_24h']:+.2f}%",
                f"#{coin2_data['market_cap_rank']}",
                f"{coin2_data['volatility_24h']:.2%}",
                f"{coin2_data['volume_marketcap_ratio']:.4f}"
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# =====================
# TAB 4: RISK ASSESSMENT - DIPERBAIKI
# =====================
with tab4:
    st.subheader("⚠️ Analisis Risiko")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Skor Risiko Pasar")
        
        if len(df_filtered) > 0:
            # Hitung risk score komposit
            volatility_risk = df_filtered['volatility_24h'].mean() * 100
            sentiment_risk = (df_filtered['price_change_percentage_24h'] < 0).mean() * 100
            risk_score = min(max((volatility_risk * 0.6 + sentiment_risk * 0.4), 0), 100)
            
            # Tentukan level risiko
            risk_level, risk_desc, risk_color = get_risk_level(risk_score)
            
            st.markdown(f"""
            <div class="risk-box" style="border-left: 5px solid {risk_color};">
                <h2>{risk_level}</h2>
                <h1>{risk_score:.0f}/100</h1>
                <p>{risk_desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar sederhana
            st.markdown(f"""
            <div style="margin-top: 20px;">
                <p><b>Tingkat Risiko:</b></p>
                <div style="background: linear-gradient(90deg, #00cc00 0%, #ffcc00 30%, #ff3333 100%); 
                            height: 20px; border-radius: 10px; margin: 10px 0;">
                    <div style="width: {risk_score}%; height: 100%; background-color: rgba(255,255,255,0.3); 
                                border-radius: 10px;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 12px;">
                    <span>🟢 Rendah</span>
                    <span>🟡 Sedang</span>
                    <span>🔴 Tinggi</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detail risk components
            st.markdown("**Komponen Risiko:**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("📊 Volatilitas", f"{volatility_risk:.1f}")
                st.progress(min(volatility_risk/50, 1.0))
            with col_b:
                st.metric("📈 Sentimen", f"{sentiment_risk:.1f}")
                st.progress(min(sentiment_risk/100, 1.0))
    
    with col2:
        st.markdown("### 📋 Koin dengan Risiko Tinggi")
        
        if len(df_filtered) > 0:
            # Identifikasi koin berisiko tinggi
            risk_criteria = (
                (df_filtered['volatility_24h'] > df_filtered['volatility_24h'].quantile(0.75)) &
                (df_filtered['price_change_percentage_24h'] < 0)
            )
            high_risk_coins = df_filtered[risk_criteria].head(5)
            
            if len(high_risk_coins) > 0:
                st.markdown("⚠️ **Koin ini memiliki risiko tinggi karena:**")
                st.markdown("- Volatilitas di atas rata-rata")
                st.markdown("- Harga sedang turun")
                
                for _, coin in high_risk_coins.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div style="background-color: #2d1e1e; padding: 10px; border-radius: 5px; 
                                    border-left: 3px solid #ff3333; margin: 5px 0;">
                            <b>{coin['symbol']}</b> - {coin['name']}
                            <div style="font-size: 12px;">
                                Volatilitas: <span style="color: orange;">{coin['volatility_24h']:.2%}</span> | 
                                Perubahan: <span style="color: red;">{coin['price_change_percentage_24h']:.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("🎉 Tidak ada koin dengan risiko tinggi dalam filter ini!")
    
    # SAFE COINS RECOMMENDATION
    st.subheader("🛡️ Rekomendasi untuk Pemula")
    
    if len(df_filtered) > 0:
        # Koin dengan risiko rendah
        safe_criteria = (
            (df_filtered['market_cap_rank'] <= 50) &  # Top 50 by market cap
            (df_filtered['volatility_24h'] < df_filtered['volatility_24h'].quantile(0.25)) &  # Low volatility
            (df_filtered['volume_marketcap_ratio'] > 0.01)  # Good liquidity
        )
        safe_coins = df_filtered[safe_criteria].head(10)
        
        if len(safe_coins) > 0:
            st.markdown("**✅ Koin yang relatif aman untuk pemula:**")
            
            # Tampilkan dalam grid
            cols = st.columns(3)
            for idx, (_, coin) in enumerate(safe_coins.iterrows()):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div style="background-color: #1e2d1e; padding: 10px; border-radius: 5px; 
                                border-left: 3px solid #33cc33; margin: 5px 0;">
                        <b>{coin['symbol']}</b>
                        <div style="font-size: 12px;">
                            Rank: #{coin['market_cap_rank']}<br>
                            Vol: {coin['volatility_24h']:.2%}<br>
                            Change: <span style="color: {'green' if coin['price_change_percentage_24h'] > 0 else 'red'}">
                            {('+' if coin['price_change_percentage_24h'] > 0 else '')}{coin['price_change_percentage_24h']:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Tidak ada koin yang memenuhi kriteria 'aman' dalam filter ini.")
    
    # RISK MANAGEMENT TIPS
    with st.expander("📚 Tips Mengelola Risiko untuk Pemula", expanded=True):
        st.markdown("""
        **🎯 Strategi untuk Pemula:**
        
        1. **Mulai dengan Koin Besar**
           - Bitcoin, Ethereum, dan koin top 20 lainnya
           - Lebih stabil, likuiditas tinggi
        
        2. **Diversifikasi Portofolio**
           - Jangan taruh semua dana di 1 koin
           - Bagikan ke 3-5 koin berbeda
        
        3. **Tentukan Batas Rugi (Stop Loss)**
           - Tentukan berapa persen kerugian yang bisa ditoleransi
           - Contoh: Jual otomatis jika turun 10%
        
        4. **Pelajari Dulu, Baru Investasi Besar**
           - Mulai dengan jumlah kecil untuk belajar
           - Pahami pola pergerakan harga
        
        5. **Hindari FOMO (Fear of Missing Out)**
           - Jangan terburu-buru ikut tren
           - Riset sebelum membeli koin baru
        """)

# =====================
# TAB 5: QUICK INSIGHTS (BARU) - UNTUK PEMULA
# =====================
with tab5:
    st.header("🚀 Panduan Cepat untuk Pemula")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 5 Hal yang Harus Diketahui")
        
        insights = []
        
        # Insight 1: Market overview
        if len(df_filtered) > 0:
            avg_change = df_filtered['price_change_percentage_24h'].mean()
            if avg_change > 2:
                insights.append("**📈 Pasar Sedang Baik**: Rata-rata koin naik hari ini")
            elif avg_change < -2:
                insights.append("**📉 Pasar Sedang Turun**: Banyak koin merah hari ini")
            else:
                insights.append("**⚖️ Pasar Stabil**: Tidak banyak perubahan berarti")
        
        # Insight 2: Best performing category
        if 'category' in df_filtered.columns:
            cat_perf = df_filtered.groupby('category')['price_change_percentage_24h'].mean()
            if not cat_perf.empty:
                best_cat = cat_perf.idxmax()
                insights.append(f"**🏆 Kategori Terbaik**: {best_cat} memberikan return terbaik")
        
        # Insight 3: Liquidity insight
        avg_vol_ratio = df_filtered['volume_marketcap_ratio'].mean() if len(df_filtered) > 0 else 0
        if avg_vol_ratio > 0.05:
            insights.append("**💎 Likuiditas Tinggi**: Mudah beli/jual koin")
        else:
            insights.append("**⚠️ Likuiditas Rendah**: Hati-hati saat transaksi besar")
        
        # Insight 4: Top coin performance
        if len(df_filtered) > 0:
            top_coin = df_filtered.nsmallest(1, 'market_cap_rank').iloc[0]
            insights.append(f"**👑 Koin Terbesar**: {top_coin['symbol']} dominasi pasar")
        
        # Insight 5: Risk level
        if len(df_filtered) > 0:
            volatility = df_filtered['volatility_24h'].mean() * 100
            if volatility < 5:
                insights.append("**🟢 Risiko Rendah**: Pasar tidak terlalu fluktuatif")
            elif volatility < 15:
                insights.append("**🟡 Risiko Normal**: Fluktuasi wajar untuk crypto")
            else:
                insights.append("**🔴 Risiko Tinggi**: Pasar sangat volatil")
        
        # Tampilkan insights
        for i, insight in enumerate(insights[:5], 1):
            st.markdown(f"{i}. {insight}")
    
    with col2:
        st.subheader("💡 5 Langkah Memulai")
        
        st.markdown("""
        <div class="tip-box">
        <ol>
        <li><b>Pilih Koin Besar</b><br>
        Mulai dari Bitcoin, Ethereum, atau koin top 10 lainnya</li><br>
        
        <li><b>Cek Volume Trading</b><br>
        Pastikan Volume/MC Ratio > 0.01</li><br>
        
        <li><b>Perhatikan Peringkat</b><br>
        Market cap rank menunjukkan ukuran dan stabilitas</li><br>
        
        <li><b>Mulai dengan Kecil</b><br>
        Investasi kecil dulu untuk belajar dan pahami pola</li><br>
        
        <li><b>Pantau Secara Rutin</b><br>
        Gunakan dashboard ini untuk monitoring harian</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # QUICK COMPARISON: CRYPTO vs TRADITIONAL ASSETS
    st.subheader("⚖️ Perbandingan: Crypto vs Aset Tradisional")
    
    comparison_data = {
        'Aset': ['Bitcoin (Crypto)', 'Saham Blue Chip', 'Emas', 'Deposito Bank'],
        'Risiko': ['Tinggi', 'Sedang', 'Rendah', 'Sangat Rendah'],
        'Potensi Return': ['Sangat Tinggi', 'Tinggi', 'Rendah', 'Sangat Rendah'],
        'Likuiditas': ['Tinggi', 'Tinggi', 'Sedang', 'Rendah'],
        'Cocok untuk': ['Investor Berpengalaman', 'Investor Umum', 'Investor Konservatif', 'Penyimpan Dana']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # PERSONALIZED RECOMMENDATION
    st.subheader("🎯 Rekomendasi Personal")
    
    experience_level = st.select_slider(
        "Tingkat Pengalaman Anda:",
        options=["Pemula Total", "Sudah Coba-Coba", "Cukup Berpengalaman", "Expert"]
    )
    
    if experience_level == "Pemula Total":
        st.success("""
        **🎯 Rekomendasi untuk Anda:**
        
        1. **Fokus pada**: Bitcoin, Ethereum, BNB (koin top 10)
        2. **Alokasi**: 70% koin besar, 30% koin mid-cap
        3. **Strategi**: Beli dan tahan (hold), hindari trading harian
        4. **Risiko**: Batasi maksimal 5-10% dari total portofolio Anda
        """)
    elif experience_level == "Sudah Coba-Coba":
        st.success("""
        **🎯 Rekomendasi untuk Anda:**
        
        1. **Fokus pada**: Koin top 50 dengan volume tinggi
        2. **Alokasi**: 50% big cap, 30% mid cap, 20% small cap pilihan
        3. **Strategi**: Bisa mulai trading swing (beberapa hari/minggu)
        4. **Risiko**: Maksimal 10-15% dari total portofolio
        """)
    elif experience_level == "Cukup Berpengalaman":
        st.success("""
        **🎯 Rekomendasi untuk Anda:**
        
        1. **Fokus pada**: Semua kategori dengan riset mendalam
        2. **Alokasi**: Sesuaikan dengan risk appetite dan analisis
        3. **Strategi**: Bisa trading harian dengan risk management ketat
        4. **Risiko**: Maksimal 20% dari total portofolio
        """)
    else:
        st.success("""
        **🎯 Anda sudah berpengalaman:**
        
        1. **Fokus pada**: Opportunity di semua segment
        2. **Alokasi**: Diversifikasi + spesialisasi niche tertentu
        3. **Strategi**: Advanced strategies dengan hedging
        4. **Risiko**: Manage sesuai risk/reward ratio target
        """)

# =====================
# FOOTER
# =====================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("📌 Dashboard Crypto untuk Pemula | Streamlit")

with footer_col2:
    st.caption(f"🔄 Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

with footer_col3:
    st.caption(f"🔍 {len(df_filtered)} koin ditampilkan")

# FINAL TIP
st.markdown("""
<div style="text-align: center; padding: 20px; background-color: #1e1e1e; border-radius: 10px; margin-top: 20px;">
    <h4>💎 Ingat: Crypto adalah aset berisiko tinggi</h4>
    <p>Selalu riset sebelum investasi, jangan investasi lebih dari yang bisa Anda tanggung kerugiannya</p>
</div>
""", unsafe_allow_html=True)
