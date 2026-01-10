import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import RobustScaler
import time
import logging
from datetime import datetime

# =====================
# SETUP LOGGING
# =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Crypto Market Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================
# CUSTOM STYLE (MINIMALIS & RESPONSIVE)
# =====================
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #0e1117;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Metric styling */
    .metric-label {
        font-size: 14px;
        color: #9aa0a6;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 8px 8px 0px 0px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3a3f4b;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e88e5;
        color: white;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1e88e5;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #1976d2;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3);
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px;
            font-size: 12px;
            height: 40px;
        }
        
        .metric-value {
            font-size: 18px;
        }
        
        h1 { font-size: 24px; }
        h2 { font-size: 20px; }
        h3 { font-size: 18px; }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #262730;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1e88e5;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #1976d2;
    }
    
    /* Card-like containers */
    .card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# =====================
# HELPER FUNCTIONS
# =====================
def create_dummy_data():
    """Create dummy data if CSV file is not found"""
    logger.warning("Creating dummy data for demonstration")
    
    np.random.seed(42)
    n_coins = 100
    
    data = {
        'id': [f'coin_{i}' for i in range(n_coins)],
        'symbol': [f'COIN{i:03d}' for i in range(n_coins)],
        'name': [f'Crypto Coin {i}' for i in range(n_coins)],
        'current_price': np.random.lognormal(3, 1.5, n_coins),
        'market_cap': np.random.lognormal(20, 1, n_coins) * 1e6,
        'market_cap_rank': np.arange(1, n_coins + 1),
        'total_volume': np.random.lognormal(18, 1, n_coins) * 1e6,
        'high_24h': np.random.lognormal(3.1, 1.5, n_coins),
        'low_24h': np.random.lognormal(2.9, 1.5, n_coins),
        'price_change_24h': np.random.normal(0, 100, n_coins),
        'price_change_percentage_24h': np.random.normal(0, 10, n_coins),
        'market_cap_change_24h': np.random.normal(0, 1e6, n_coins),
        'market_cap_change_percentage_24h': np.random.normal(0, 5, n_coins),
        'fully_diluted_valuation': np.random.lognormal(21, 1, n_coins) * 1e6,
        'circulating_supply': np.random.lognormal(16, 1, n_coins),
        'total_supply': np.random.lognormal(16.5, 1, n_coins),
        'max_supply': np.random.lognormal(17, 1, n_coins),
        'ath': np.random.lognormal(4, 1, n_coins),
        'ath_change_percentage': np.random.uniform(-80, 100, n_coins),
        'ath_date': [datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ') for _ in range(n_coins)],
        'last_updated': [datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ') for _ in range(n_coins)]
    }
    
    return pd.DataFrame(data)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """
    Load crypto data from CSV file with error handling
    Returns: DataFrame with crypto data
    """
    try:
        logger.info("Attempting to load data from CSV file...")
        df = pd.read_csv("crypto_top1000_dataset.csv")
        
        if df.empty:
            logger.warning("CSV file is empty. Creating dummy data.")
            df = create_dummy_data()
            st.sidebar.warning("⚠️ Menggunakan data dummy karena file kosong")
        else:
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
        return df
        
    except FileNotFoundError:
        logger.error("CSV file not found. Creating dummy data.")
        st.sidebar.error("❌ File 'crypto_top1000_dataset.csv' tidak ditemukan")
        return create_dummy_data()
        
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty or corrupted.")
        st.sidebar.error("❌ File CSV kosong atau korup")
        return create_dummy_data()
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.sidebar.error(f"❌ Error memuat data: {str(e)}")
        return create_dummy_data()

def validate_data(df):
    """
    Validate required columns and data quality
    Returns: Tuple (is_valid, message)
    """
    required_columns = [
        'market_cap', 'current_price', 'price_change_percentage_24h',
        'high_24h', 'low_24h', 'total_volume', 'market_cap_rank'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Kolom penting tidak ditemukan: {missing_columns}"
    
    # Check for null values in critical columns
    critical_nulls = df[required_columns].isnull().sum().sum()
    if critical_nulls > 0:
        logger.warning(f"Found {critical_nulls} null values in critical columns")
    
    return True, "Data valid"

def categorize(rank):
    """Categorize coins by market cap rank"""
    if rank <= 10: 
        return "Big Cap"
    if rank <= 50: 
        return "Mid Cap"
    if rank <= 200: 
        return "Small Cap"
    return "Micro Cap"

def cap_outliers(series):
    """Cap outliers using IQR method"""
    if len(series) < 4:
        return series
    
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    
    if iqr == 0:  # Handle constant series
        return series
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return series.clip(lower=lower_bound, upper=upper_bound)

def calculate_risk_score(df_subset):
    """Calculate comprehensive risk score for coins"""
    risk_factors = pd.Series(0, index=df_subset.index)
    
    if 'volatility_24h' in df_subset.columns:
        if df_subset['volatility_24h'].std() > 0:
            volatility_norm = (df_subset['volatility_24h'] - df_subset['volatility_24h'].min()) / \
                             (df_subset['volatility_24h'].max() - df_subset['volatility_24h'].min())
            risk_factors += volatility_norm.fillna(0) * 0.4
    
    if 'price_change_percentage_24h' in df_subset.columns:
        # Negative returns indicate higher risk
        negative_returns = (df_subset['price_change_percentage_24h'] < 0).astype(float)
        risk_factors += negative_returns * 0.3
    
    if 'volume_marketcap_ratio' in df_subset.columns:
        if df_subset['volume_marketcap_ratio'].std() > 0:
            # Low volume relative to market cap is risky
            volume_risk = 1 - ((df_subset['volume_marketcap_ratio'] - df_subset['volume_marketcap_ratio'].min()) / \
                             (df_subset['volume_marketcap_ratio'].max() - df_subset['volume_marketcap_ratio'].min()))
            risk_factors += volume_risk.fillna(0) * 0.3
    
    return risk_factors

# =====================
# LOAD & PREPARE DATA
# =====================
with st.spinner('🔄 Memuat data cryptocurrency...'):
    df = load_data()
    time.sleep(0.5)  # Brief pause for better UX

# Validate data
is_valid, validation_msg = validate_data(df)
if not is_valid:
    st.error(f"⚠️ {validation_msg}")
    st.info("Menggunakan data dummy untuk melanjutkan...")
    df = create_dummy_data()

# =====================
# DATA CLEANING & FEATURE ENGINEERING
# =====================
logger.info("Cleaning and engineering features...")

# Drop rows with critical missing values
critical_cols = ["high_24h", "low_24h", "price_change_24h", 
                 "price_change_percentage_24h", "market_cap_change_24h"]
df_clean = df.dropna(subset=[col for col in critical_cols if col in df.columns])

# Create new features
df_clean = df_clean.copy()

# Basic features
df_clean["volatility_24h"] = (df_clean["high_24h"] - df_clean["low_24h"]) / df_clean["current_price"].replace(0, np.nan)
df_clean["volume_marketcap_ratio"] = df_clean["total_volume"] / df_clean["market_cap"].replace(0, np.nan)

# Supply features
if "max_supply" in df_clean.columns and "circulating_supply" in df_clean.columns:
    df_clean["max_supply_available"] = df_clean["max_supply"].notna().astype(int)
    df_clean["supply_utilization"] = df_clean["circulating_supply"] / df_clean["max_supply"].replace(0, np.nan)
    df_clean["supply_inflation_risk"] = 1 - df_clean["supply_utilization"].fillna(0).clip(0, 1)
else:
    df_clean["supply_inflation_risk"] = 0.5  # Default moderate risk

# Valuation features
if "fully_diluted_valuation" in df_clean.columns:
    df_clean["fdv_available"] = df_clean["fully_diluted_valuation"].notna().astype(int)
    df_clean["fdv_mc_ratio"] = df_clean["fully_diluted_valuation"] / df_clean["market_cap"].replace(0, np.nan)

# Handle outliers for visualization
scale_cols = ["market_cap", "total_volume", "volatility_24h", "volume_marketcap_ratio"]
scale_cols = [col for col in scale_cols if col in df_clean.columns]

# Save original for reference
for col in scale_cols:
    df_clean[f"{col}_original"] = df_clean[col].copy()

# Cap outliers
for col in scale_cols:
    df_clean[col] = cap_outliers(df_clean[col])

# Scale features
if scale_cols:
    scaler = RobustScaler()
    df_clean[scale_cols] = scaler.fit_transform(df_clean[scale_cols])

# Normalize size for bubble charts
if "total_volume_original" in df_clean.columns:
    vol_min = df_clean["total_volume_original"].min()
    vol_max = df_clean["total_volume_original"].max()
    
    if vol_max > vol_min:
        df_clean["size_normalized"] = ((df_clean["total_volume_original"] - vol_min) / 
                                      (vol_max - vol_min) * 30 + 5)
    else:
        df_clean["size_normalized"] = 15  # Default size

# Add category
df_clean["category"] = df_clean["market_cap_rank"].apply(categorize)

# Calculate risk scores
df_clean["risk_score"] = calculate_risk_score(df_clean)

logger.info(f"Data preparation complete. Final shape: {df_clean.shape}")

# =====================
# SIDEBAR
# =====================
with st.sidebar:
    st.title("⚙️ Kontrol Dashboard")
    
    # Quick guide
    with st.expander("📖 Panduan Cepat", expanded=False):
        st.markdown("""
        **Glossary:**
        - **Market Cap**: Nilai total pasar = harga × jumlah koin beredar
        - **Volatilitas**: Ukuran fluktuasi harga
        - **FDV/MC Ratio**: Perbandingan nilai penuh vs nilai pasar saat ini
        - **Volume/MC Ratio**: Aktivitas trading relatif terhadap ukuran pasar
        - **Risk Score**: Skor risiko 0-1 (semakin tinggi = semakin berisiko)
        """)
    
    st.markdown("---")
    st.subheader("🔍 Filter Data")
    
    # Market cap rank filter
    rank_range = st.slider(
        "Market Cap Rank",
        1, min(1000, len(df_clean)),
        (1, 100),
        help="Filter berdasarkan peringkat market cap (1 = terbesar)"
    )
    
    # Performance filter
    st.markdown("---")
    st.subheader("🎯 Filter Performa")
    
    performance_filter = st.selectbox(
        "Tampilkan koin dengan:",
        ["Semua Koin", "Harga Naik 24h", "Harga Turun 24h", "Volatilitas Tinggi", 
         "Volume Trading Tinggi", "Risiko Tinggi", "Risiko Rendah"],
        help="Filter berdasarkan performa atau karakteristik koin"
    )
    
    # Category filter
    st.markdown("---")
    st.subheader("🏷️ Filter Kategori")
    
    categories = df_clean["category"].unique()
    selected_categories = st.multiselect(
        "Pilih Kategori Market Cap:",
        options=categories,
        default=categories,
        help="Pilih satu atau lebih kategori"
    )
    
    # Price range filter
    st.markdown("---")
    st.subheader("💰 Filter Harga")
    
    if 'current_price' in df_clean.columns:
        price_min = float(df_clean['current_price'].min())
        price_max = float(df_clean['current_price'].max())
        
        price_range = st.slider(
            "Rentang Harga (USD)",
            min_value=price_min,
            max_value=price_max,
            value=(price_min, min(price_max, 1000)),
            step=0.1,
            format="%.2f"
        )
    
    # Theme selector
    st.markdown("---")
    st.subheader("🎨 Tema Visualisasi")
    
    theme = st.selectbox(
        "Pilih Tema Plotly:",
        ["plotly_dark", "plotly", "plotly_white", "ggplot2", "seaborn"],
        index=0,
        help="Pilih tema warna untuk chart"
    )
    
    # Apply theme
    px.defaults.template = theme
    
    # Export data
    st.markdown("---")
    st.subheader("💾 Export Data")
    
    if st.button("📥 Download Filtered Data", use_container_width=True):
        st.session_state['export_data'] = True

# =====================
# APPLY FILTERS
# =====================
df_filtered = df_clean.copy()

# Apply rank filter
df_filtered = df_filtered[
    (df_filtered["market_cap_rank"] >= rank_range[0]) &
    (df_filtered["market_cap_rank"] <= rank_range[1])
]

# Apply category filter
if selected_categories:
    df_filtered = df_filtered[df_filtered["category"].isin(selected_categories)]

# Apply price filter
if 'current_price' in df_filtered.columns and 'price_range' in locals():
    df_filtered = df_filtered[
        (df_filtered["current_price"] >= price_range[0]) &
        (df_filtered["current_price"] <= price_range[1])
    ]

# Apply performance filter
if performance_filter == "Harga Naik 24h":
    df_filtered = df_filtered[df_filtered["price_change_percentage_24h"] > 0]
elif performance_filter == "Harga Turun 24h":
    df_filtered = df_filtered[df_filtered["price_change_percentage_24h"] < 0]
elif performance_filter == "Volatilitas Tinggi":
    if 'volatility_24h' in df_filtered.columns:
        threshold = df_filtered["volatility_24h"].quantile(0.75)
        df_filtered = df_filtered[df_filtered["volatility_24h"] > threshold]
elif performance_filter == "Volume Trading Tinggi":
    if 'volume_marketcap_ratio' in df_filtered.columns:
        threshold = df_filtered["volume_marketcap_ratio"].quantile(0.75)
        df_filtered = df_filtered[df_filtered["volume_marketcap_ratio"] > threshold]
elif performance_filter == "Risiko Tinggi":
    threshold = df_filtered["risk_score"].quantile(0.75)
    df_filtered = df_filtered[df_filtered["risk_score"] > threshold]
elif performance_filter == "Risiko Rendah":
    threshold = df_filtered["risk_score"].quantile(0.25)
    df_filtered = df_filtered[df_filtered["risk_score"] < threshold]

# Store in session state for export
st.session_state['df_filtered'] = df_filtered

# =====================
# HEADER
# =====================
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("📊 Crypto Market Dashboard")
    st.markdown("""
    <div style='color: #9aa0a6; font-size: 16px;'>
    Analisis <b>market dominance, volatilitas, dan performa harga kripto</b> secara interaktif
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("Total Koin", len(df_filtered))
    
with col3:
    if len(df_filtered) > 0 and 'price_change_percentage_24h' in df_filtered.columns:
        avg_change = df_filtered['price_change_percentage_24h'].mean()
        st.metric("Avg Change 24h", f"{avg_change:+.2f}%")

# Export data if requested
if 'export_data' in st.session_state and st.session_state['export_data']:
    csv = df_filtered.to_csv(index=False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    st.sidebar.download_button(
        label="💾 Download CSV",
        data=csv,
        file_name=f"crypto_data_{timestamp}.csv",
        mime="text/csv",
        key='download_csv'
    )
    st.session_state['export_data'] = False

# =====================
# TABS FOR ORGANIZATION
# =====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview", 
    "🏆 Top Performers", 
    "📊 Detail Analisis", 
    "⚠️ Risk Assessment"
])

# =====================
# TAB 1: OVERVIEW
# =====================
with tab1:
    # KPI METRICS
    st.subheader("📊 Market Snapshot")
    
    # Row 1: Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_market_cap = df_filtered['market_cap'].sum() if 'market_cap' in df_filtered.columns else 0
        st.metric("Total Market Cap", f"${total_market_cap:,.0f}")
        st.caption("Nilai pasar total")
    
    with col2:
        if len(df_filtered) > 0 and 'price_change_percentage_24h' in df_filtered.columns:
            gainers = (df_filtered["price_change_percentage_24h"] > 0).sum()
            total_coins = len(df_filtered)
            gainer_percentage = (gainers / total_coins * 100) if total_coins > 0 else 0
            st.metric("Koin Naik (24h)", f"{gainers}/{total_coins}", 
                     f"{gainer_percentage:.1f}%")
        else:
            st.metric("Koin Naik (24h)", "N/A")
    
    with col3:
        if 'volatility_24h' in df_filtered.columns:
            avg_vol = df_filtered["volatility_24h"].mean()
            st.metric("Avg Volatility", f"{avg_vol:.2%}")
            st.caption("24 jam terakhir")
        else:
            st.metric("Avg Volatility", "N/A")
    
    with col4:
        if 'risk_score' in df_filtered.columns:
            avg_risk = df_filtered["risk_score"].mean()
            st.metric("Avg Risk Score", f"{avg_risk:.2f}")
            st.caption("0-1 (higher = riskier)")
        else:
            st.metric("Avg Risk Score", "N/A")
    
    # Row 2: Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if 'volume_marketcap_ratio' in df_filtered.columns:
            avg_volume_ratio = df_filtered["volume_marketcap_ratio"].mean()
            st.metric("Avg Volume/MC", f"{avg_volume_ratio:.3f}")
            st.caption("Rasio aktivitas")
    
    with col6:
        if 'category' in df_filtered.columns:
            big_cap_count = (df_filtered["category"] == "Big Cap").sum()
            st.metric("Big Cap Coins", big_cap_count)
            st.caption("Rank 1-10")
    
    with col7:
        if 'fdv_mc_ratio' in df_filtered.columns:
            avg_fdv_ratio = df_filtered['fdv_mc_ratio'].median()
            st.metric("Median FDV/MC", f"{avg_fdv_ratio:.2f}")
            st.caption("Potensi pengenceran")
    
    with col8:
        if 'price_change_percentage_24h' in df_filtered.columns:
            median_return = df_filtered['price_change_percentage_24h'].median()
            st.metric("Median Return 24h", f"{median_return:+.2f}%")
            st.caption("Return median")
    
    st.markdown("---")
    
    # ROW 1: Market Dominance & Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗺️ Dominasi Market")
        
        if len(df_filtered) > 0:
            # Limit to top 30 for clarity
            display_df = df_filtered.nsmallest(30, "market_cap_rank")
            
            if not display_df.empty:
                fig = px.treemap(
                    display_df,
                    path=["category", "symbol"],
                    values="market_cap",
                    color="price_change_percentage_24h",
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                    hover_data={
                        "current_price": ":.2f",
                        "market_cap_rank": True,
                        "price_change_percentage_24h": ":.2f%"
                    },
                    title="Market Dominance (Top 30 by Market Cap)"
                )
                fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Ukuran: Market Cap | Warna: Perubahan harga 24h")
            else:
                st.info("Tidak ada data untuk ditampilkan")
        else:
            st.info("Tidak ada data yang sesuai dengan filter")
    
    with col2:
        st.subheader("📊 Distribusi Perubahan Harga")
        
        if len(df_filtered) > 0 and 'price_change_percentage_24h' in df_filtered.columns:
            fig = px.histogram(
                df_filtered,
                x="price_change_percentage_24h",
                nbins=30,
                color_discrete_sequence=['#1e88e5'],
                title="Distribusi Return 24 Jam"
            )
            fig.add_vline(x=0, line_dash="dash", line_color="white", 
                         annotation_text="Netral", annotation_position="top")
            
            fig.update_layout(
                xaxis_title="Perubahan Harga (%) - 24h",
                yaxis_title="Jumlah Koin",
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Distribusi perubahan harga dalam 24 jam terakhir")
        else:
            st.info("Data perubahan harga tidak tersedia")
    
    # ROW 2: Market Health Indicators
    st.subheader("❤️ Market Health Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Market Sentiment Gauge
        if len(df_filtered) > 0 and 'price_change_percentage_24h' in df_filtered.columns:
            sentiment_score = df_filtered['price_change_percentage_24h'].mean()
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sentiment_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sentimen Pasar", 'font': {'size': 16}},
                delta={'reference': 0},
                gauge={
                    'axis': {'range': [-20, 20], 'tickwidth': 1},
                    'bar': {'color': "#1e88e5"},
                    'steps': [
                        {'range': [-20, -5], 'color': "#ef5350"},
                        {'range': [-5, 5], 'color': "#ffca28"},
                        {'range': [5, 20], 'color': "#66bb6a"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.85,
                        'value': sentiment_score
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada data sentimen")
    
    with col2:
        # Volume Health Gauge
        if len(df_filtered) > 0 and 'volume_marketcap_ratio' in df_filtered.columns:
            volume_health = df_filtered['volume_marketcap_ratio'].mean() * 100
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=min(volume_health, 10),  # Cap at 10 for visualization
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Kesehatan Volume", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 10]},
                    'bar': {'color': "#43a047"},
                    'steps': [
                        {'range': [0, 3], 'color': "#ef5350"},
                        {'range': [3, 7], 'color': "#ffca28"},
                        {'range': [7, 10], 'color': "#66bb6a"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.85,
                        'value': min(volume_health, 10)
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Data volume tidak tersedia")
    
    with col3:
        # Market Cap Distribution
        if len(df_filtered) > 0 and 'category' in df_filtered.columns:
            category_counts = df_filtered['category'].value_counts()
            
            if not category_counts.empty:
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    hole=0.5,
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    title="Distribusi Kapitalisasi"
                )
                fig.update_layout(
                    height=250,
                    margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Data kategori tidak tersedia")
        else:
            st.info("Tidak ada data yang sesuai")

# =====================
# TAB 2: TOP PERFORMERS
# =====================
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Market Cap")
        
        # Get top 10 by market cap
        top_10 = df_clean.nsmallest(10, "market_cap_rank")
        
        if not top_10.empty:
            # Create display dataframe
            display_cols = ['symbol', 'name', 'market_cap', 'current_price', 
                          'price_change_percentage_24h', 'category']
            display_cols = [col for col in display_cols if col in top_10.columns]
            
            display_df = top_10[display_cols].copy()
            
            # Format for display
            if 'market_cap' in display_df.columns:
                display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"${x:,.0f}")
            
            if 'current_price' in display_df.columns:
                display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
            
            if 'price_change_percentage_24h' in display_df.columns:
                display_df['price_change_percentage_24h'] = display_df['price_change_percentage_24h'].apply(
                    lambda x: f"{'+' if x > 0 else ''}{x:.2f}%")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "symbol": "Symbol",
                    "name": "Name",
                    "market_cap": "Market Cap",
                    "current_price": "Price",
                    "price_change_percentage_24h": "24h Change",
                    "category": "Category"
                }
            )
            
            # Bar chart visualization
            fig = px.bar(
                top_10,
                x="symbol",
                y="market_cap",
                color="price_change_percentage_24h",
                text_auto=".2s",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                title="Top 10 by Market Cap"
            )
            fig.update_layout(
                xaxis_title="Symbol",
                yaxis_title="Market Cap",
                coloraxis_colorbar=dict(title="24h Change %")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada data top 10")
    
    with col2:
        st.subheader("📈 Top Gainers vs Losers (24h)")
        
        if len(df_filtered) > 0 and 'price_change_percentage_24h' in df_filtered.columns:
            # Get top 5 gainers and losers
            top_gainers = df_filtered.nlargest(5, "price_change_percentage_24h")
            top_losers = df_filtered.nsmallest(5, "price_change_percentage_24h")
            
            # Combine and mark
            top_gainers['type'] = 'Gainer'
            top_losers['type'] = 'Loser'
            combo = pd.concat([top_gainers, top_losers])
            
            # Display table
            display_cols = ['symbol', 'name', 'price_change_percentage_24h', 
                          'current_price', 'market_cap', 'type']
            display_cols = [col for col in display_cols if col in combo.columns]
            
            display_df = combo[display_cols].copy()
            
            # Format for display
            if 'price_change_percentage_24h' in display_df.columns:
                display_df['price_change_percentage_24h'] = display_df['price_change_percentage_24h'].apply(
                    lambda x: f"{'+' if x > 0 else ''}{x:.2f}%")
            
            if 'current_price' in display_df.columns:
                display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
            
            # Apply color coding
            def color_type(val):
                if val == 'Gainer':
                    return 'color: #4caf50'
                else:
                    return 'color: #f44336'
            
            styled_df = display_df.style.applymap(color_type, subset=['type'])
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "symbol": "Symbol",
                    "name": "Name",
                    "price_change_percentage_24h": "24h Change",
                    "current_price": "Price",
                    "market_cap": "Market Cap",
                    "type": "Type"
                }
            )
            
            # Visualization
            fig = px.bar(
                combo,
                x="price_change_percentage_24h",
                y="symbol",
                orientation="h",
                color="type",
                color_discrete_map={'Gainer': '#4caf50', 'Loser': '#f44336'},
                title="Top Gainers vs Losers"
            )
            fig.update_layout(
                xaxis_title="Price Change (%)",
                yaxis_title="Symbol",
                legend_title="Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada data untuk gainers/losers")
    
    # Heatmap Performance
    st.subheader("🎨 Heatmap Performa - Top 20 Koin")
    
    if len(df_filtered) > 0:
        # Select top 20 by market cap within filtered data
        top_20 = df_filtered.nsmallest(20, "market_cap_rank")
        
        if not top_20.empty:
            # Define metrics for heatmap
            metric_config = {
                'price_change_percentage_24h': '24h Return',
                'price_change_percentage_7d': '7d Return',
                'volatility_24h': 'Volatility',
                'volume_marketcap_ratio': 'Volume/MC',
                'risk_score': 'Risk Score',
                'fdv_mc_ratio': 'FDV/MC Ratio'
            }
            
            # Check which metrics are available
            available_metrics = {k: v for k, v in metric_config.items() 
                               if k in top_20.columns and not top_20[k].isna().all()}
            
            if available_metrics:
                # Prepare data for heatmap
                heatmap_data = top_20.set_index('symbol')[list(available_metrics.keys())]
                heatmap_data.columns = [available_metrics[col] for col in heatmap_data.columns]
                
                # Create heatmap
                fig = px.imshow(
                    heatmap_data.T,
                    color_continuous_scale="RdYlGn",
                    aspect="auto",
                    title="Performance Heatmap (Top 20 by Market Cap)"
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Cryptocurrency",
                    yaxis_title="Metric"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation
                with st.expander("ℹ️ Interpretasi Heatmap"):
                    st.markdown("""
                    **Warna pada heatmap:**
                    - **Hijau**: Nilai baik (return tinggi, risiko rendah)
                    - **Kuning**: Nilai sedang
                    - **Merah**: Nilai buruk (return negatif, risiko tinggi)
                    
                    **Metrik:**
                    - **24h/7d Return**: Persentase perubahan harga
                    - **Volatility**: Tingkat fluktuasi harga
                    - **Volume/MC**: Rasio volume trading terhadap market cap
                    - **Risk Score**: Skor risiko komposit (0-1)
                    - **FDV/MC Ratio**: Rasio valuasi penuh terhadap market cap
                    """)
            else:
                st.info("Metrik performa tidak tersedia untuk heatmap")
        else:
            st.info("Tidak cukup data untuk heatmap")
    else:
        st.info("Tidak ada data yang sesuai dengan filter")

# =====================
# TAB 3: DETAIL ANALISIS
# =====================
with tab3:
    st.subheader("💰 Analisis Harga vs Market Cap")
    
    if len(df_filtered) > 0:
        # Use top 100 for clarity
        scatter_data = df_filtered.nsmallest(100, "market_cap_rank").copy()
        
        # Create scatter plot
        fig = px.scatter(
            scatter_data,
            x="current_price",
            y="market_cap",
            log_x=True,
            log_y=True,
            color="category",
            size="size_normalized",
            hover_name="name",
            hover_data={
                "symbol": True,
                "price_change_percentage_24h": ":.2f%",
                "volatility_24h": ":.3f",
                "volume_marketcap_ratio": ":.3f",
                "market_cap_rank": True,
                "category": False
            },
            title="Harga vs Market Cap (Log Scale)",
            labels={
                "current_price": "Current Price (USD, log)",
                "market_cap": "Market Cap (log)",
                "category": "Category"
            }
        )
        
        fig.update_layout(
            hovermode="closest",
            legend=dict(
                title="Market Cap Category",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_traces(
            marker=dict(
                opacity=0.8,
                line=dict(width=1, color='DarkSlateGrey')
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("""
        **Interpretasi:**
        - **Ukuran bubble**: Volume trading (semakin besar = volume lebih tinggi)
        - **Warna**: Kategori market cap
        - **Posisi**: Harga (sumbu X) vs Market Cap (sumbu Y)
        """)
    else:
        st.info("Tidak ada data yang sesuai dengan filter")
    
    # Correlation Matrix
    st.subheader("🔗 Matriks Korelasi")
    
    if len(df_filtered) > 0:
        # Define numeric columns for correlation
        numeric_cols = [
            'current_price', 'market_cap', 'total_volume',
            'price_change_percentage_24h', 'volatility_24h',
            'volume_marketcap_ratio', 'risk_score'
        ]
        
        # Check which columns are available
        available_numeric = [col for col in numeric_cols 
                           if col in df_filtered.columns and df_filtered[col].dtype in ['int64', 'float64']]
        
        if len(available_numeric) >= 3:  # Need at least 3 for meaningful correlation
            # Calculate correlation
            corr_matrix = df_filtered[available_numeric].corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                text_auto=".2f",
                title="Korelasi antar Variabel",
                labels=dict(color="Korelasi")
            )
            
            fig.update_layout(
                height=500,
                xaxis_title="Variable",
                yaxis_title="Variable"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation guide
            with st.expander("📖 Panduan Interpretasi Korelasi"):
                st.markdown("""
                **Nilai Korelasi:**
                - **+1.0**: Korelasi positif sempurna (bergerak searah)
                - **+0.7 sampai +1.0**: Korelasi positif kuat
                - **+0.3 sampai +0.7**: Korelasi positif moderat
                - **-0.3 sampai +0.3**: Tidak ada korelasi signifikan
                - **-0.3 sampai -0.7**: Korelasi negatif moderat
                - **-0.7 sampai -1.0**: Korelasi negatif kuat
                - **-1.0**: Korelasi negatif sempurna (bergerak berlawanan)
                
                **Insights yang berguna:**
                - Market Cap vs Volume: Biasanya berkorelasi positif
                - Volatility vs Risk Score: Biasanya berkorelasi positif
                - Price Change vs Volume: Korelasi bisa positif atau negatif
                """)
        else:
            st.info("Tidak cukup data numerik untuk analisis korelasi")
    else:
        st.info("Tidak ada data yang sesuai dengan filter")
    
    # Detailed Data Table
    st.subheader("📋 Detail Data Kripto")
    
    if len(df_filtered) > 0:
        # Let user select columns to display
        available_columns = [
            'name', 'symbol', 'current_price', 'price_change_percentage_24h',
            'market_cap', 'market_cap_rank', 'total_volume', 'volatility_24h',
            'volume_marketcap_ratio', 'risk_score', 'category'
        ]
        
        available_columns = [col for col in available_columns if col in df_filtered.columns]
        
        selected_columns = st.multiselect(
            "Pilih kolom untuk ditampilkan:",
            options=available_columns,
            default=['symbol', 'name', 'current_price', 'price_change_percentage_24h', 'market_cap', 'category']
        )
        
        # Number of rows to show
        num_rows = st.slider("Jumlah baris:", 10, 100, 20)
        
        if selected_columns:
            display_data = df_filtered[selected_columns].head(num_rows).copy()
            
            # Format numeric columns
            if 'current_price' in display_data.columns:
                display_data['current_price'] = display_data['current_price'].apply(
                    lambda x: f"${x:,.2f}")
            
            if 'price_change_percentage_24h' in display_data.columns:
                display_data['price_change_percentage_24h'] = display_data['price_change_percentage_24h'].apply(
                    lambda x: f"{'+' if x > 0 else ''}{x:.2f}%")
            
            if 'market_cap' in display_data.columns:
                display_data['market_cap'] = display_data['market_cap'].apply(
                    lambda x: f"${x:,.0f}")
            
            if 'total_volume' in display_data.columns:
                display_data['total_volume'] = display_data['total_volume'].apply(
                    lambda x: f"${x:,.0f}")
            
            if 'volatility_24h' in display_data.columns:
                display_data['volatility_24h'] = display_data['volatility_24h'].apply(
                    lambda x: f"{x:.2%}")
            
            if 'risk_score' in display_data.columns:
                display_data['risk_score'] = display_data['risk_score'].apply(
                    lambda x: f"{x:.3f}")
            
            # Display dataframe
            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )
            
            # Add search functionality
            st.markdown("---")
            st.subheader("🔍 Pencarian Koin Spesifik")
            
            search_term = st.text_input("Cari berdasarkan nama atau symbol:")
            
            if search_term:
                search_results = df_filtered[
                    df_filtered['name'].str.contains(search_term, case=False, na=False) |
                    df_filtered['symbol'].str.contains(search_term, case=False, na=False)
                ]
                
                if not search_results.empty:
                    st.write(f"Ditemukan {len(search_results)} hasil:")
                    st.dataframe(search_results[selected_columns].head(10), use_container_width=True)
                else:
                    st.info("Tidak ditemukan koin yang sesuai dengan pencarian")
        else:
            st.info("Silakan pilih kolom untuk ditampilkan")
    else:
        st.info("Tidak ada data yang sesuai dengan filter")

# =====================
# TAB 4: RISK ASSESSMENT
# =====================
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Indikator Risiko Pasar")
        
        if len(df_filtered) > 0:
            # Calculate market risk indicators
            risk_indicators = {}
            
            # Volatility risk (40%)
            if 'volatility_24h' in df_filtered.columns:
                volatility_risk = df_filtered['volatility_24h'].mean() * 100
                risk_indicators['Volatility'] = min(volatility_risk, 100)
            else:
                risk_indicators['Volatility'] = 50  # Default moderate
            
            # Sentiment risk (30%)
            if 'price_change_percentage_24h' in df_filtered.columns:
                sentiment_risk = (df_filtered['price_change_percentage_24h'] < 0).mean() * 100
                risk_indicators['Sentiment'] = sentiment_risk
            else:
                risk_indicators['Sentiment'] = 50
            
            # Concentration risk (15%)
            if 'market_cap' in df_filtered.columns and df_filtered['market_cap'].sum() > 0:
                top_10_mc = df_filtered.nsmallest(10, 'market_cap_rank')['market_cap'].sum()
                total_mc = df_filtered['market_cap'].sum()
                concentration = (top_10_mc / total_mc) * 100
                risk_indicators['Concentration'] = min(concentration, 100)
            else:
                risk_indicators['Concentration'] = 50
            
            # Volume risk (15%)
            if 'volume_marketcap_ratio' in df_filtered.columns:
                low_volume_ratio = (df_filtered['volume_marketcap_ratio'] < 0.01).mean() * 100
                risk_indicators['Liquidity'] = low_volume_ratio
            else:
                risk_indicators['Liquidity'] = 50
            
            # Calculate composite risk score
            weights = {'Volatility': 0.4, 'Sentiment': 0.3, 'Concentration': 0.15, 'Liquidity': 0.15}
            market_risk_score = sum(risk_indicators[k] * weights[k] for k in weights.keys())
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=market_risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Market Risk Score", 'font': {'size': 20}},
                delta={'reference': 50, 'increasing': {'color': "red"}, 
                      'decreasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#1e88e5"},
                    'steps': [
                        {'range': [0, 30], 'color': "#66bb6a"},
                        {'range': [30, 70], 'color': "#ffca28"},
                        {'range': [70, 100], 'color': "#ef5350"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.85,
                        'value': market_risk_score
                    }
                }
            ))
            
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk breakdown
            st.markdown("**Detail Risiko Pasar:**")
            
            risk_cols = st.columns(4)
            risk_items = list(risk_indicators.items())
            
            for idx, (name, value) in enumerate(risk_items):
                with risk_cols[idx % 4]:
                    st.metric(name, f"{value:.1f}")
            
            # Risk interpretation
            risk_level = "RENDAH" if market_risk_score < 30 else \
                        "SEDANG" if market_risk_score < 70 else "TINGGI"
            
            risk_color = "#66bb6a" if risk_level == "RENDAH" else \
                        "#ffca28" if risk_level == "SEDANG" else "#ef5350"
            
            st.markdown(f"""
            <div style='background-color: {risk_color}20; padding: 15px; border-radius: 10px; border-left: 4px solid {risk_color}; margin: 10px 0;'>
            <h4 style='color: {risk_color}; margin-top: 0;'>Tingkat Risiko: <b>{risk_level}</b></h4>
            <p style='margin-bottom: 0;'>
            {f"Pasar relatif stabil dengan risiko rendah" if risk_level == "RENDAH" else 
              f"Pasar dengan volatilitas moderat" if risk_level == "SEDANG" else 
              f"Pasar berisiko tinggi dengan volatilitas tinggi"}
            </p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.info("Tidak ada data untuk kalkulasi risiko pasar")
    
    with col2:
        st.subheader("📊 Profil Risiko per Kategori")
        
        if len(df_filtered) > 0 and 'category' in df_filtered.columns:
            # Calculate risk metrics by category
            risk_by_category = []
            
            categories = df_filtered['category'].unique()
            
            for cat in categories:
                cat_data = df_filtered[df_filtered['category'] == cat]
                
                if len(cat_data) > 0:
                    cat_metrics = {'Category': cat}
                    
                    # Average volatility
                    if 'volatility_24h' in cat_data.columns:
                        cat_metrics['Volatility'] = cat_data['volatility_24h'].mean()
                    
                    # Percent with negative returns
                    if 'price_change_percentage_24h' in cat_data.columns:
                        cat_metrics['Negative Returns'] = (cat_data['price_change_percentage_24h'] < 0).mean()
                    
                    # Average risk score
                    if 'risk_score' in cat_data.columns:
                        cat_metrics['Risk Score'] = cat_data['risk_score'].mean()
                    
                    # Average volume ratio
                    if 'volume_marketcap_ratio' in cat_data.columns:
                        cat_metrics['Liquidity'] = cat_data['volume_marketcap_ratio'].mean()
                    
                    risk_by_category.append(cat_metrics)
            
            if risk_by_category:
                risk_df = pd.DataFrame(risk_by_category)
                
                # Create radar chart
                metrics = [col for col in risk_df.columns if col != 'Category']
                
                if len(metrics) >= 3:  # Need at least 3 metrics for radar
                    fig = go.Figure()
                    
                    for _, row in risk_df.iterrows():
                        values = [row[metric] for metric in metrics]
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=metrics,
                            fill='toself',
                            name=row['Category']
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=True,
                        title="Risk Profile by Category",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback to bar chart
                    melted_df = risk_df.melt(id_vars=['Category'], var_name='Metric', value_name='Value')
                    
                    fig = px.bar(
                        melted_df,
                        x='Category',
                        y='Value',
                        color='Metric',
                        barmode='group',
                        title="Risk Metrics by Category"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tidak cukup data untuk analisis risiko per kategori")
        else:
            st.info("Data kategori tidak tersedia")
    
    # High Risk Coins Table
    st.subheader("🔴 Koin dengan Risiko Tertinggi")
    
    if len(df_filtered) > 0 and 'risk_score' in df_filtered.columns:
        # Get top 10 highest risk coins
        high_risk = df_filtered.nlargest(10, 'risk_score').copy()
        
        # Select columns for display
        display_cols = ['symbol', 'name', 'risk_score', 'volatility_24h', 
                       'price_change_percentage_24h', 'volume_marketcap_ratio', 'category']
        display_cols = [col for col in display_cols if col in high_risk.columns]
        
        display_df = high_risk[display_cols].copy()
        
        # Format values
        if 'risk_score' in display_df.columns:
            display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.3f}")
        
        if 'volatility_24h' in display_df.columns:
            display_df['volatility_24h'] = display_df['volatility_24h'].apply(lambda x: f"{x:.2%}")
        
        if 'price_change_percentage_24h' in display_df.columns:
            display_df['price_change_percentage_24h'] = display_df['price_change_percentage_24h'].apply(
                lambda x: f"{'+' if x > 0 else ''}{x:.2f}%")
        
        if 'volume_marketcap_ratio' in display_df.columns:
            display_df['volume_marketcap_ratio'] = display_df['volume_marketcap_ratio'].apply(lambda x: f"{x:.4f}")
        
        # Apply color coding based on risk score
        def highlight_risk(val):
            try:
                risk_val = float(val)
                if risk_val > 0.7:
                    return 'background-color: #ffcccc'
                elif risk_val > 0.5:
                    return 'background-color: #fff3cd'
                else:
                    return ''
            except:
                return ''
        
        styled_df = display_df.style.applymap(highlight_risk, subset=['risk_score'])
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "symbol": "Symbol",
                "name": "Name",
                "risk_score": "Risk Score",
                "volatility_24h": "Volatility",
                "price_change_percentage_24h": "24h Change",
                "volume_marketcap_ratio": "Volume/MC Ratio",
                "category": "Category"
            }
        )
        
        # Risk mitigation suggestions
        with st.expander("🛡️ Strategi Mitigasi Risiko"):
            st.markdown("""
            **Untuk koin dengan risiko tinggi:**
            
            1. **Position Sizing**: Alokasikan porsi portfolio yang lebih kecil
            2. **Stop Loss**: Gunakan stop loss order untuk membatasi kerugian
            3. **Diversifikasi**: Seimbangkan dengan aset berisiko lebih rendah
            4. **Monitoring**: Pantau lebih sering untuk perubahan kondisi
            5. **Fundamental Analysis**: Periksa apakah ada masalah fundamental
            
            **Faktor risiko yang diperhitungkan:**
            - **Volatilitas tinggi**: Fluktuasi harga yang ekstrem
            - **Return negatif**: Tekanan jual yang konsisten
            - **Likuiditas rendah**: Volume trading kecil relatif terhadap market cap
            - **Kategori kecil**: Small/micro cap umumnya lebih berisiko
            """)
    else:
        st.info("Data risiko tidak tersedia")
    
    # Risk vs Return Analysis
    st.subheader("📈 Analisis Risk vs Return")
    
    if len(df_filtered) > 0 and 'risk_score' in df_filtered.columns and 'price_change_percentage_24h' in df_filtered.columns:
        # Create scatter plot of risk vs return
        scatter_data = df_filtered.copy()
        
        # Limit for clarity
        if len(scatter_data) > 100:
            scatter_data = scatter_data.nsmallest(100, 'market_cap_rank')
        
        fig = px.scatter(
            scatter_data,
            x='risk_score',
            y='price_change_percentage_24h',
            color='category',
            size='market_cap',
            hover_name='name',
            hover_data={
                'symbol': True,
                'volatility_24h': ':.3f',
                'volume_marketcap_ratio': ':.4f'
            },
            title='Risk vs Return Analysis',
            labels={
                'risk_score': 'Risk Score',
                'price_change_percentage_24h': '24h Return (%)',
                'category': 'Market Cap Category'
            }
        )
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.add_vline(x=0.5, line_dash="dash", line_color="white", opacity=0.5)
        
        # Add quadrant annotations
        fig.add_annotation(x=0.25, y=10, text="Low Risk\nHigh Return", showarrow=False, font=dict(color="green"))
        fig.add_annotation(x=0.75, y=10, text="High Risk\nHigh Return", showarrow=False, font=dict(color="orange"))
        fig.add_annotation(x=0.25, y=-10, text="Low Risk\nLow Return", showarrow=False, font=dict(color="blue"))
        fig.add_annotation(x=0.75, y=-10, text="High Risk\nLow Return", showarrow=False, font=dict(color="red"))
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("""
        **Interpretasi Kuadran:**
        - **Kiri Atas (Ideal)**: Risiko rendah, return tinggi
        - **Kanan Atas**: Risiko tinggi, return tinggi (high risk-high reward)
        - **Kiri Bawah**: Risiko rendah, return rendah (safe but low growth)
        - **Kanan Bawah (Hindari)**: Risiko tinggi, return rendah (worst scenario)
        """)

# =====================
# FOOTER
# =====================
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("📊 Crypto Market Dashboard v2.0")
    st.caption("Streamlit + Plotly + Pandas")

with footer_col2:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"🕐 Terakhir diperbarui: {current_time}")

with footer_col3:
    st.caption(f"📈 Data points: {len(df_filtered)} koin")
    st.caption(f"🗂️ Dataset: {len(df_clean)} total koin")

# =====================
# DEBUG INFO (Collapsed)
# =====================
with st.expander("🔧 Debug Information", expanded=False):
    st.subheader("Data Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Data:**")
        st.write(f"- Shape: {df.shape}")
        st.write(f"- Columns: {len(df.columns)}")
        st.write(f"- Memory: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    
    with col2:
        st.write("**Filtered Data:**")
        st.write(f"- Shape: {df_filtered.shape}")
        st.write(f"- Range: Rank {rank_range[0]} to {rank_range[1]}")
        st.write(f"- Categories: {', '.join(df_filtered['category'].unique())}")
    
    st.subheader("Column Information")
    
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()
    
    st.write(f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}")
    st.write(f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:10])}{'...' if len(categorical_cols) > 10 else ''}")
    
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.rerun()

# Add a little JavaScript for better UX
st.markdown("""
<script>
// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Add loading state to buttons
document.querySelectorAll('.stButton button').forEach(button => {
    button.addEventListener('click', function() {
        this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    });
});
</script>
""", unsafe_allow_html=True)
