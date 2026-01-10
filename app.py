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

# ===== RAW FEATURES (UNTUK LOGIKA & RISK) =====
df["volatility_24h_raw"] = (df["high_24h"] - df["low_24h"]) / df["current_price"]
df["volume_marketcap_ratio_raw"] = df["total_volume"] / df["market_cap"]

# Supply inflation risk HARUS dibatasi
df["supply_inflation_risk_raw"] = (
    1 - df["supply_utilization"]
).clip(lower=0, upper=1)

# =====================
# OUTLIER CAPPING & SCALING (PERBAIKAN UNTUK SIZE)
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
    "volatility_24h_raw",
    "volume_marketcap_ratio_raw"
]

# Simpan data asli untuk visualisasi
df["total_volume_original"] = df["total_volume"].copy()

for col in scale_cols:
    df[col] = cap_outliers(df[col])

scaler = RobustScaler()
scaled_values = scaler.fit_transform(df[scale_cols])

df["market_cap_scaled"] = scaled_values[:, 0]
df["total_volume_scaled"] = scaled_values[:, 1]
df["volatility_24h_scaled"] = scaled_values[:, 2]
df["volume_marketcap_ratio_scaled"] = scaled_values[:, 3]

# Normalisasi size untuk visualisasi (pastikan tidak negatif)
df["size_normalized"] = (df["total_volume_original"] - df["total_volume_original"].min()) / \
                        (df["total_volume_original"].max() - df["total_volume_original"].min()) * 30 + 5

# =====================
# HELPER FUNCTIONS
# =====================
def categorize(rank):
    if rank <= 10: return "Big Cap"
    if rank <= 50: return "Mid Cap"
    return "Small Cap"

# =====================
# SIDEBAR
# =====================
st.sidebar.title("⚙️ Filter & Kontrol")

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
    "Market Cap Rank",
    1, 1000, (1, 100),
    help="Filter berdasarkan peringkat market cap (1 = terbesar)"
)

# Filter tambahan untuk pemula
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Filter Performa")

performance_filter = st.sidebar.selectbox(
    "Tampilkan koin dengan:",
    ["Semua Koin", "Harga Naik 24h", "Harga Turun 24h", "Volatilitas Tinggi", "Volume Trading Tinggi"],
    help="Filter berdasarkan performa koin"
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
elif performance_filter == "Volatilitas Tinggi":
    df_filtered = df_filtered[
        df_filtered["volatility_24h_raw"] >
        df_filtered["volatility_24h_raw"].quantile(0.75)
    ]
elif performance_filter == "Volume Trading Tinggi":
    df_filtered = df_filtered[df_filtered["volume_marketcap_ratio"] > df_filtered["volume_marketcap_ratio"].quantile(0.75)]

# Kategori market cap
df_filtered["category"] = df_filtered["market_cap_rank"].apply(categorize)

# =====================
# HEADER
# =====================
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📊 Crypto Market Dashboard")
    st.markdown(
        "Analisis **market dominance, volatilitas, dan performa harga kripto** secara interaktif."
    )
with col2:
    st.info("ℹ️ Dashboard interaktif untuk analisis pasar kripto")

# =====================
# TABS FOR ORGANIZATION
# =====================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🏆 Top Performers", "📊 Detail Analisis", "⚠️ Risk Assessment"])

# =====================
# TAB 1: OVERVIEW
# =====================
with tab1:
    # KPI METRICS
    st.subheader("📊 Market Snapshot")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Koin", len(df_filtered))
        st.caption("Dalam range yang dipilih")
    
    with col2:
        avg_vol = df_filtered["volatility_24h"].mean()
        st.metric("Rata-rata Volatilitas", f"{avg_vol:.2%}")
        st.caption("24 jam terakhir")
    
    with col3:
        avg_volume_ratio = df_filtered["volume_marketcap_ratio"].mean()
        st.metric("Avg Volume/MarketCap", f"{avg_volume_ratio:.3f}")
        st.caption("Rasio aktivitas")
    
    with col4:
        gainers = (df_filtered["price_change_percentage_24h"] > 0).sum()
        total_coins = len(df_filtered)
        st.metric("Koin Naik (24h)", f"{gainers}/{total_coins}", 
                 f"{(gainers/total_coins*100):.1f}%")
    
    # Additional KPIs
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        total_market_cap = df_filtered['market_cap'].sum()
        st.metric("Total Market Cap", f"${total_market_cap:,.0f}")
        st.caption("Nilai pasar total")
    
    with col6:
        top_10 = df.nsmallest(10, "market_cap_rank")
        dominance = (top_10['market_cap'].sum() / df_filtered['market_cap'].sum() * 100) if df_filtered['market_cap'].sum() > 0 else 0
        st.metric("Dominasi Top 10", f"{dominance:.1f}%")
        st.caption("Konsentrasi pasar")
    
    with col7:
        gain_ratio = (df_filtered['price_change_percentage_24h'] > 0).mean()
        st.metric("Rasio Naik/Turun", f"{gain_ratio:.1%}")
        st.caption("Sentimen pasar")
    
    with col8:
        avg_fdv_ratio = df_filtered['fdv_mc_ratio'].median()
        st.metric("Avg FDV/MC Ratio", f"{avg_fdv_ratio:.2f}")
        st.caption("Potensi pengenceran")
    
    st.markdown("---")
    
    # ROW 1: Market Dominance & Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗺️ Dominasi Market")
        fig = px.treemap(
            df_filtered.head(30),  # Batasi untuk visual yang lebih jelas
            path=["category", "symbol"],
            values="market_cap",
            color="price_change_percentage_24h",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            template="plotly_dark",
            hover_data=["current_price", "market_cap_rank"]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Ukuran: Market Cap | Warna: Perubahan harga 24h")
    
    with col2:
        st.subheader("📊 Distribusi Volatilitas")
        fig = px.histogram(
            df_filtered,
            x="price_change_percentage_24h",
            nbins=30,
            template="plotly_dark",
            color_discrete_sequence=['#1e88e5']
        )
        fig.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="Netral")
        fig.update_layout(
            xaxis_title="Perubahan Harga (%) - 24h",
            yaxis_title="Jumlah Koin"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Distribusi perubahan harga dalam 24 jam terakhir")
    
    # ROW 2: Market Health Indicators
    st.subheader("❤️ Market Health Indicators")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Market Sentiment Gauge
        sentiment_score = df_filtered['price_change_percentage_24h'].mean() * 100 if len(df_filtered) > 0 else 0
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentimen Pasar"},
            gauge={
                'axis': {'range': [-20, 20]},
                'bar': {'color': "darkblue"},
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
        fig.update_layout(template="plotly_dark", height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Volume Health
        volume_health = df_filtered['volume_marketcap_ratio'].mean() * 100 if len(df_filtered) > 0 else 0
        volume_health = min(max(volume_health, 0), 10)  # Batasi antara 0-10
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=volume_health,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Kesehatan Volume"},
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
        fig.update_layout(template="plotly_dark", height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Risk Distribution
        if len(df_filtered) > 0:
            risk_counts = df_filtered['category'].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                hole=0.5,
                color_discrete_sequence=px.colors.sequential.RdBu,
                template="plotly_dark"
            )
            fig.update_layout(
                title="Distribusi Kapitalisasi",
                showlegend=True,
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada data yang sesuai dengan filter")

# =====================
# TAB 2: TOP PERFORMERS
# =====================
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Market Cap")
        top_10 = df.nsmallest(10, "market_cap_rank")
        
        # Format untuk display
        display_df = top_10[['symbol', 'name', 'market_cap', 'current_price', 
                           'price_change_percentage_24h']].copy()
        display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"${x:,.0f}")
        display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
        display_df['price_change_percentage_24h'] = display_df['price_change_percentage_24h'].apply(
            lambda x: f"{'+' if x > 0 else ''}{x:.2f}%")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Visual bar chart
        fig = px.bar(
            top_10,
            x="symbol",
            y="market_cap",
            color="price_change_percentage_24h",
            text_auto=".2s",
            color_continuous_scale="RdYlGn",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Top Gainers vs Losers (24 Jam)")
        
        if len(df_filtered) > 0:
            # Top gainers
            top_5 = df_filtered.nlargest(5, "price_change_percentage_24h")
            bot_5 = df_filtered.nsmallest(5, "price_change_percentage_24h")
            combo = pd.concat([top_5, bot_5])
            
            # Display table
            gainers_df = combo[['symbol', 'name', 'price_change_percentage_24h', 
                              'current_price', 'market_cap']].copy()
            gainers_df['price_change_percentage_24h'] = gainers_df['price_change_percentage_24h'].apply(
                lambda x: f"{'+' if x > 0 else ''}{x:.2f}%")
            gainers_df['current_price'] = gainers_df['current_price'].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(gainers_df, use_container_width=True, hide_index=True)
            
            # Visual chart
            fig = px.bar(
                combo,
                x="price_change_percentage_24h",
                y="symbol",
                orientation="h",
                color="price_change_percentage_24h",
                color_continuous_scale="RdYlGn",
                template="plotly_dark",
                labels={"price_change_percentage_24h": "Perubahan Harga (%)"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada data yang sesuai dengan filter")
    
    # Heatmap Performance
    st.subheader("🎨 Heatmap Performa - Top 20 Koin")
    
    if len(df_filtered) > 0:
        # Pilih 20 koin terbesar dalam filter
        top_20 = df_filtered.nsmallest(20, "market_cap_rank")
        
        # Buat matriks performa
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
                'volume_marketcap_ratio': 'Volume/MC Ratio',
                'fdv_mc_ratio': 'FDV/MC Ratio'
            }
            performance_df.columns = [metric_names.get(col, col) for col in available_metrics]
            
            fig = px.imshow(performance_df.T,
                            color_continuous_scale="RdYlGn",
                            aspect="auto",
                            template="plotly_dark",
                            labels=dict(x="Kripto", y="Metrik", color="Nilai"))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Metrik performa tidak tersedia")
    else:
        st.info("Tidak ada data yang sesuai dengan filter")

# =====================
# TAB 3: DETAIL ANALISIS
# =====================
with tab3:
    st.subheader("💰 Analisis Harga vs Market Cap")
    
    if len(df_filtered) > 0:
        # Scatter plot dengan kategori
        scatter_data = df_filtered.head(100).copy()  # Batasi untuk visual yang lebih jelas
        
        fig = px.scatter(
            scatter_data,
            x="current_price",
            y="market_cap",
            log_x=True,
            log_y=True,
            color="category",
            size="size_normalized",   # ✅ CUKUP SATU
            hover_data={
                "price_change_percentage_24h": ":.2f%",
                "volatility_24h_raw": ":.2%",
                "volume_marketcap_ratio_raw": ":.3f",
                "market_cap_rank": True
            },
            template="plotly_dark",
            labels={
                "current_price": "Harga (log)",
                "market_cap": "Market Cap (log)",
                "category": "Kategori"
            }
        )
        
        # Tambah trendline
        fig.update_traces(marker=dict(opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Ukuran bubble: Volume Trading | Warna: Kategori Market Cap")
    else:
        st.info("Tidak ada data yang sesuai dengan filter")
    
    # Correlation Matrix
    st.subheader("🔗 Matriks Korelasi")
    
    if len(df_filtered) > 0:
        numeric_cols = ['current_price', 'market_cap', 'total_volume', 
                       'price_change_percentage_24h', 'volatility_24h',
                       'volume_marketcap_ratio', 'fdv_mc_ratio']
        
        # Hanya ambil kolom yang ada
        available_numeric = [col for col in numeric_cols if col in df_filtered.columns]
        
        if len(available_numeric) > 1:
            corr_df = df_filtered[available_numeric].corr()
            
            fig = px.imshow(corr_df,
                            color_continuous_scale="RdBu",
                            zmin=-1, zmax=1,
                            text_auto=".2f",
                            template="plotly_dark",
                            title="Korelasi antar Variabel")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak cukup data numerik untuk korelasi")
    else:
        st.info("Tidak ada data yang sesuai dengan filter")
    
    # Summary Table
    st.subheader("📋 Ringkasan Data")
    
    if len(df_filtered) > 0:
        summary_cols = ['name', 'symbol', 'current_price', 'price_change_percentage_24h',
                       'market_cap', 'volatility_24h', 'category']
        
        # Hanya ambil kolom yang ada
        available_summary = [col for col in summary_cols if col in df_filtered.columns]
        
        summary_df = df_filtered.head(20)[available_summary].copy()
        
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
        
        # Color code the price change
        def color_price_change(val):
            if isinstance(val, str) and '+' in val:
                return 'color: green'
            elif isinstance(val, str) and '-' in val:
                return 'color: red'
            return ''
        
        if 'price_change_percentage_24h' in summary_df.columns:
            styled_df = summary_df.style.applymap(color_price_change, 
                                                  subset=['price_change_percentage_24h'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("Tidak ada data yang sesuai dengan filter")

# =====================
# TAB 4: RISK ASSESSMENT
# =====================
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Indikator Risiko Pasar")
        
        # Hitung risk score komposit
        if len(df_filtered) > 0:
            volatility_risk = (
                df_filtered["volatility_24h_raw"]
                .clip(0, df_filtered["volatility_24h_raw"].quantile(0.95))
                .mean() * 100
            )
            
            sentiment_risk = (
                df_filtered["price_change_percentage_24h"] < 0
            ).mean() * 100
            
            inflation_risk = (
                df_filtered["supply_inflation_risk_raw"]
                .mean() * 100
            )
            
            risk_score = (
                volatility_risk * 0.4 +
                sentiment_risk * 0.3 +
                inflation_risk * 0.3
            )
            
            risk_score = np.clip(risk_score, 0, 100)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Tingkat Risiko Pasar", 'font': {'size': 20}},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
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
            
            # Risk breakdown
            st.markdown("**Breakdown Risiko:**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Volatilitas", f"{volatility_risk:.1f}")
            with col_b:
                st.metric("Sentimen", f"{sentiment_risk:.1f}")
            with col_c:
                st.metric("Inflation", f"{inflation_risk:.1f}")
        else:
            st.info("Tidak ada data untuk kalkulasi risiko")
    
    with col2:
        st.subheader("📊 Distribusi Risiko per Kategori")
        
        if len(df_filtered) > 0 and 'category' in df_filtered.columns:
            # Hitung risk metrics per kategori
            risk_metrics = df_filtered.groupby("category").agg({
                "volatility_24h_raw": "mean",
                "price_change_percentage_24h": lambda x: (x < 0).mean(),
                "supply_inflation_risk_raw": "mean"
            }).reset_index()

            risk_metrics.rename(columns={
                "volatility_24h_raw": "Volatility Risk",
                "price_change_percentage_24h": "Negative Sentiment",
                "supply_inflation_risk_raw": "Supply Inflation Risk"
            }, inplace=True)
            
            # Hapus kolom yang tidak ada
            risk_metrics = risk_metrics.dropna(axis=1, how='all')
            
            if len(risk_metrics) > 0:
                # Normalisasi untuk radar chart
                numeric_cols = risk_metrics.select_dtypes(include=[np.number]).columns.tolist()
                for col in numeric_cols:
                    min_val = risk_metrics[col].min()
                    max_val = risk_metrics[col].max()
                    if max_val > min_val:  # Hindari pembagian dengan nol
                        risk_metrics[col] = (risk_metrics[col] - min_val) / (max_val - min_val)
                
                # Buat radar chart
                categories = risk_metrics['category'].tolist()
                metrics = [col for col in numeric_cols]
                
                fig = go.Figure()
                
                for idx, row in risk_metrics.iterrows():
                    values = [row[col] for col in metrics]
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metrics,
                        fill='toself',
                        name=row['category']
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tidak cukup data untuk radar chart")
        else:
            st.info("Data kategori tidak tersedia")
    
    # High Risk Coins
    st.subheader("🔴 Koin dengan Risiko Tinggi")
    
    if len(df_filtered) > 0:
        # Hitung risk score untuk setiap koin
        df_filtered["risk_score_temp"] = (
            df_filtered["volatility_24h_raw"].rank(pct=True) * 0.4 +
            (df_filtered["price_change_percentage_24h"] < 0).astype(int) * 0.3 +
            df_filtered["supply_inflation_risk_raw"].rank(pct=True) * 0.3
        )
        
        # Tambah supply inflation risk jika ada
        df_filtered['risk_score_temp'] += (
            df_filtered['supply_inflation_risk_raw'].rank(pct=True) * 0.3
        )
        
        high_risk = df_filtered.nlargest(10, 'risk_score_temp')[['symbol', 'name', 'risk_score_temp', 
                                                               'volatility_24h', 
                                                               'price_change_percentage_24h',
                                                               'category']].copy()
        
        high_risk['risk_score'] = high_risk['risk_score_temp'].apply(lambda x: f"{x:.1%}")
        high_risk['volatility_24h'] = high_risk['volatility_24h'].apply(lambda x: f"{x:.2%}")
        high_risk['price_change_percentage_24h'] = high_risk['price_change_percentage_24h'].apply(
            lambda x: f"{'+' if x > 0 else ''}{x:.2f}%")
        
        # Hapus kolom temporary
        high_risk = high_risk.drop('risk_score_temp', axis=1)
        
        st.dataframe(high_risk, use_container_width=True, hide_index=True)
    else:
        st.info("Tidak ada data untuk analisis risiko")
    
    # Risk Factors Explanation
    with st.expander("📖 Faktor Risiko yang Diperhitungkan"):
        st.markdown("""
        **1. Volatilitas (40%)**
        - Mengukur fluktuasi harga dalam 24 jam
        - Semakin tinggi volatilitas, semakin tinggi risiko
        
        **2. Sentimen Pasar (30%)**
        - Proporsi koin dengan harga turun dalam 24 jam
        - Indikator tekanan jual di pasar
        
        **3. Risiko Inflasi (30%)**
        - Potensi penambahan supply koin
        - Semakin tinggi supply inflation risk, semakin tinggi risiko pengenceran
        """)

# =====================
# FOOTER
# =====================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("📌 Data Visualization Project | Streamlit Dashboard")

with footer_col2:
    st.caption(f"🔄 Data diperbarui: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

with footer_col3:
    st.caption(f"🔍 Total data point: {len(df_filtered)}")


