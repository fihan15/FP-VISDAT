import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# =====================
# PAGE CONFIG & THEME
# =====================
st.set_page_config(
    page_title="📊 Crypto Market Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# =====================
# CUSTOM STYLING - DARK MODE
# =====================
st.markdown("""
<style>
    /* Reset untuk dark mode compatibility */
    .main {
        background-color: #0e1117;
    }
    
    .stApp {
        background-color: #0e1117;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #f0f2f6 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1a202c;
    }
    
    .sidebar .sidebar-content {
        background-color: #1a202c;
    }
    
    /* Title styling */
    .main-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #00d4ff, #0083ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem !important;
        padding: 20px 0;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.1rem !important;
        color: #a0aec0 !important;
        text-align: center;
        margin-bottom: 2rem !important;
        font-weight: 300;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border-left: 4px solid #00d4ff;
        margin-bottom: 12px;
        border: 1px solid #2d3748;
    }
    
    /* Chart container styling */
    .chart-container {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #2d3748;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        color: #00d4ff !important;
        margin-top: 25px !important;
        margin-bottom: 15px !important;
        padding-bottom: 8px;
        border-bottom: 2px solid #00d4ff;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(90deg, #00d4ff, #0083ff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: #00d4ff !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: #1a202c !important;
        color: #f0f2f6 !important;
        border: 1px solid #2d3748 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Processing status boxes */
    .processing-box {
        background: #1a202c;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #00d4ff;
        border: 1px solid #2d3748;
    }
    
    .processing-success {
        border-left-color: #00ff88;
    }
    
    .processing-warning {
        border-left-color: #ffaa00;
    }
    
    .processing-error {
        border-left-color: #ff5555;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    """Load data dari file CSV"""
    try:
        df = pd.read_csv("crypto_top1000_dataset.csv")
        
        # Jika file tidak ada, buat sample data
        if df.empty:
            raise FileNotFoundError
        
        return df
    except:
        # Create sample data jika file tidak ditemukan
        st.warning("⚠️ Using sample data for presentation")
        np.random.seed(42)
        n = 200
        data = {
            'name': [f'Crypto{i}' for i in range(n)],
            'symbol': [f'CRYPTO{i:03d}' for i in range(n)],
            'market_cap': np.random.exponential(10000, n) * 1e6,
            'current_price': np.random.lognormal(0, 2, n),
            'price_change_percentage_24h': np.random.normal(2, 15, n),
            'price_change_percentage_7d': np.random.normal(5, 20, n),
            'total_volume': np.random.exponential(1000, n) * 1e6,
            'market_cap_rank': range(1, n+1),
            'high_24h': np.random.uniform(1.05, 1.15, n),
            'low_24h': np.random.uniform(0.90, 0.98, n),
            'circulating_supply': np.random.exponential(1e6, n),
            'max_supply': np.random.exponential(2e6, n),
        }
        df = pd.DataFrame(data)
        return df

df_raw = load_data()

# =====================
# SIDEBAR FILTERS & PREPROCESSING CONTROLS
# =====================
with st.sidebar:
    st.markdown('<h2 style="color: #00d4ff;">⚙️ Dashboard Controls</h2>', unsafe_allow_html=True)
    
    # Data Processing Options
    with st.expander("🔧 Data Processing Settings", expanded=True):
        st.markdown("### Preprocessing Steps")
        
        # Missing Value Handling Options
        missing_method = st.selectbox(
            "Missing Value Handling:",
            ["Drop rows with critical missing", "Impute with median", "Keep as is"]
        )
        
        # Outlier Handling Options
        outlier_method = st.selectbox(
            "Outlier Handling:",
            ["IQR Capping (1.5*IQR)", "Z-Score (±3σ)", "Percentile (1-99%)", "None"]
        )
        
        # Scaling Options
        scaling_method = st.selectbox(
            "Feature Scaling:",
            ["RobustScaler", "MinMaxScaler (0-1)", "StandardScaler", "None"]
        )
        
        # Show processing stats
        show_stats = st.checkbox("Show processing statistics", value=True)
    
    st.markdown("---")
    
    # Data Filtering
    with st.expander("🔍 Data Filtering", expanded=True):
        st.markdown("### Filter by Market Cap Rank")
        
        max_rank = min(1000, len(df_raw))
        rank_range = st.slider(
            "Select Market Cap Rank Range:",
            min_value=1,
            max_value=max_rank,
            value=(1, min(100, max_rank)),
            help="Filter cryptocurrencies by their market capitalization ranking"
        )
        
        # Additional filters
        st.markdown("### Additional Filters")
        
        price_change_filter = st.selectbox(
            "Price Change (24h):",
            ["All", "Positive only", "Negative only", "> 5% gain", "< -5% loss"]
        )
        
        market_cap_filter = st.selectbox(
            "Market Cap Category:",
            ["All", "Large Cap (Top 10)", "Mid Cap (11-50)", "Small Cap (51+)"]
        )

# =====================
# COMPREHENSIVE DATA PREPROCESSING
# =====================
def apply_preprocessing(df, missing_method, outlier_method, scaling_method, show_stats):
    """Apply comprehensive data preprocessing pipeline"""
    
    processing_log = []
    df_processed = df.copy()
    
    # 1. MISSING VALUE HANDLING
    processing_log.append("🔹 **Missing Value Handling**")
    
    original_rows = len(df_processed)
    original_missing = df_processed.isnull().sum().sum()
    
    if missing_method == "Drop rows with critical missing":
        critical_cols = ['name', 'symbol', 'market_cap', 'current_price', 'market_cap_rank']
        critical_cols = [col for col in critical_cols if col in df_processed.columns]
        df_processed = df_processed.dropna(subset=critical_cols)
        rows_dropped = original_rows - len(df_processed)
        processing_log.append(f"   - Dropped {rows_dropped} rows with critical missing values")
        
    elif missing_method == "Impute with median":
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
                processing_log.append(f"   - Imputed {col} with median: {median_val:.2f}")
    
    final_missing = df_processed.isnull().sum().sum()
    processing_log.append(f"   - Missing values: {original_missing} → {final_missing}")
    
    # 2. OUTLIER HANDLING
    processing_log.append("🔹 **Outlier Handling**")
    
    if outlier_method != "None":
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        outlier_cols = ['market_cap', 'total_volume', 'current_price', 'price_change_percentage_24h']
        outlier_cols = [col for col in outlier_cols if col in numeric_cols]
        
        for col in outlier_cols:
            if outlier_method == "IQR Capping (1.5*IQR)":
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
            elif outlier_method == "Z-Score (±3σ)":
                mean = df_processed[col].mean()
                std = df_processed[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
            elif outlier_method == "Percentile (1-99%)":
                lower_bound = df_processed[col].quantile(0.01)
                upper_bound = df_processed[col].quantile(0.99)
            
            # Cap outliers
            outliers_before = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()
            df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
            outliers_after = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()
            
            if outliers_before > 0:
                processing_log.append(f"   - {col}: Capped {outliers_before} outliers")
    
    # 3. FEATURE SCALING
    processing_log.append("🔹 **Feature Scaling**")
    
    if scaling_method != "None":
        scale_cols = ['market_cap', 'total_volume', 'current_price']
        scale_cols = [col for col in scale_cols if col in df_processed.columns]
        
        if scaling_method == "RobustScaler":
            scaler = RobustScaler()
            df_processed[scale_cols] = scaler.fit_transform(df_processed[scale_cols])
            processing_log.append(f"   - Applied RobustScaler to {len(scale_cols)} columns")
            
        elif scaling_method == "MinMaxScaler (0-1)":
            scaler = MinMaxScaler()
            df_processed[scale_cols] = scaler.fit_transform(df_processed[scale_cols])
            processing_log.append(f"   - Applied MinMaxScaler to {len(scale_cols)} columns")
            
        elif scaling_method == "StandardScaler":
            for col in scale_cols:
                df_processed[col] = (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()
            processing_log.append(f"   - Applied StandardScaler to {len(scale_cols)} columns")
    
    # 4. FEATURE ENGINEERING
    processing_log.append("🔹 **Feature Engineering**")
    
    # Calculate volatility
    if all(col in df_processed.columns for col in ['high_24h', 'low_24h', 'current_price']):
        df_processed['volatility_24h'] = (df_processed['high_24h'] - df_processed['low_24h']) / df_processed['current_price']
        processing_log.append("   - Added: volatility_24h")
    
    # Calculate volume to market cap ratio
    if all(col in df_processed.columns for col in ['total_volume', 'market_cap']):
        df_processed['volume_marketcap_ratio'] = df_processed['total_volume'] / df_processed['market_cap']
        processing_log.append("   - Added: volume_marketcap_ratio")
    
    # Create market cap categories
    if 'market_cap_rank' in df_processed.columns:
        df_processed['market_cap_category'] = pd.cut(
            df_processed['market_cap_rank'],
            bins=[0, 10, 50, 1000],
            labels=['Large Cap', 'Mid Cap', 'Small Cap']
        )
        processing_log.append("   - Added: market_cap_category")
    
    # Calculate risk score
    risk_factors = []
    if 'volatility_24h' in df_processed.columns:
        risk_factors.append(df_processed['volatility_24h'].rank(pct=True) * 0.4)
    
    if 'price_change_percentage_24h' in df_processed.columns:
        risk_factors.append((df_processed['price_change_percentage_24h'] < 0).astype(float) * 0.3)
    
    if 'market_cap_rank' in df_processed.columns:
        risk_factors.append((1 - df_processed['market_cap_rank'].rank(pct=True)) * 0.3)
    
    if risk_factors:
        df_processed['risk_score'] = sum(risk_factors) / len(risk_factors)
        processing_log.append("   - Added: risk_score")
    
    # Calculate performance score
    perf_factors = []
    if 'price_change_percentage_24h' in df_processed.columns:
        perf_factors.append(df_processed['price_change_percentage_24h'].rank(pct=True) * 0.5)
    
    if 'price_change_percentage_7d' in df_processed.columns:
        perf_factors.append(df_processed['price_change_percentage_7d'].rank(pct=True) * 0.3)
    
    if 'volume_marketcap_ratio' in df_processed.columns:
        perf_factors.append(df_processed['volume_marketcap_ratio'].rank(pct=True) * 0.2)
    
    if perf_factors:
        df_processed['performance_score'] = sum(perf_factors) / len(perf_factors)
        processing_log.append("   - Added: performance_score")
    
    # Format for display
    if 'market_cap' in df_processed.columns:
        df_processed['market_cap_formatted'] = df_processed['market_cap'].apply(
            lambda x: f'${x/1e9:.2f}B' if x >= 1e9 else f'${x/1e6:.1f}M'
        )
    
    processing_log.append(f"✅ **Processing complete**: {len(df_processed)} rows, {len(df_processed.columns)} columns")
    
    return df_processed, processing_log

# Apply preprocessing
df_processed, processing_log = apply_preprocessing(
    df_raw, missing_method, outlier_method, scaling_method, show_stats
)

# =====================
# APPLY FILTERS
# =====================
df_filtered = df_processed.copy()

# Apply market cap rank filter
if 'market_cap_rank' in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered['market_cap_rank'] >= rank_range[0]) &
        (df_filtered['market_cap_rank'] <= rank_range[1])
    ]

# Apply price change filter
if price_change_filter == "Positive only" and 'price_change_percentage_24h' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['price_change_percentage_24h'] > 0]
elif price_change_filter == "Negative only" and 'price_change_percentage_24h' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['price_change_percentage_24h'] < 0]
elif price_change_filter == "> 5% gain" and 'price_change_percentage_24h' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['price_change_percentage_24h'] > 5]
elif price_change_filter == "< -5% loss" and 'price_change_percentage_24h' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['price_change_percentage_24h'] < -5]

# Apply market cap category filter
if market_cap_filter != "All" and 'market_cap_category' in df_filtered.columns:
    if market_cap_filter == "Large Cap (Top 10)":
        df_filtered = df_filtered[df_filtered['market_cap_category'] == 'Large Cap']
    elif market_cap_filter == "Mid Cap (11-50)":
        df_filtered = df_filtered[df_filtered['market_cap_category'] == 'Mid Cap']
    elif market_cap_filter == "Small Cap (51+)":
        df_filtered = df_filtered[df_filtered['market_cap_category'] == 'Small Cap']

# =====================
# MAIN DASHBOARD
# =====================
# Header Section
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="main-title">📊 Crypto Market Intelligence Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Analytics with Comprehensive Data Preprocessing</p>', unsafe_allow_html=True)
with col2:
    st.info(f"🔍 **Showing:** {len(df_filtered)}/{len(df_raw)} coins")

# Show Processing Log if enabled
if show_stats:
    with st.expander("📊 Data Processing Summary", expanded=False):
        for log_entry in processing_log:
            st.write(log_entry)

# Key Metrics
st.markdown("---")
st.markdown('<h2 class="section-header">📈 Market Overview</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_coins = len(df_filtered)
        st.metric("Total Coins", f"{total_coins}")
        st.caption(f"Rank {rank_range[0]}-{rank_range[1]}")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'market_cap' in df_filtered.columns:
            total_market_cap = df_filtered['market_cap'].sum()
            st.metric("Total Market Cap", f"${total_market_cap/1e12:.2f}T")
        else:
            st.metric("Market Cap", "N/A")
        st.caption("Filtered range")
        st.markdown('</div>', unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'price_change_percentage_24h' in df_filtered.columns:
            avg_change = df_filtered['price_change_percentage_24h'].mean()
            st.metric("Avg 24h Change", f"{avg_change:+.1f}%")
        else:
            st.metric("24h Change", "N/A")
        st.caption("Daily performance")
        st.markdown('</div>', unsafe_allow_html=True)

with col4:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'price_change_percentage_24h' in df_filtered.columns:
            gainers = (df_filtered['price_change_percentage_24h'] > 0).sum()
            st.metric("Bullish Coins", f"{gainers}/{total_coins}")
        else:
            st.metric("Bullish", "N/A")
        st.caption(f"{(gainers/total_coins*100):.1f}% positive")
        st.markdown('</div>', unsafe_allow_html=True)

# Visualization Section
st.markdown("---")
st.markdown('<h2 class="section-header">📊 Market Analysis</h2>', unsafe_allow_html=True)

# Row 1: Market Distribution
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 🏆 Top 10 by Market Cap")
    
    if len(df_filtered) > 0 and 'market_cap' in df_filtered.columns:
        top_10 = df_filtered.nsmallest(10, 'market_cap_rank') if 'market_cap_rank' in df_filtered.columns else df_filtered.nlargest(10, 'market_cap')
        
        fig = px.bar(
            top_10,
            x='symbol',
            y='market_cap',
            color='price_change_percentage_24h' if 'price_change_percentage_24h' in top_10.columns else None,
            color_continuous_scale='RdYlGn',
            text_auto='.2s'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(26, 32, 44, 0.8)',
            paper_bgcolor='rgba(26, 32, 44, 0.8)',
            font={'color': '#f0f2f6'},
            xaxis_title="",
            yaxis_title="Market Cap"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for visualization")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📈 Price Performance")
    
    if len(df_filtered) > 5:
        # Get top 5 gainers and losers
        if 'price_change_percentage_24h' in df_filtered.columns:
            top_gainers = df_filtered.nlargest(5, 'price_change_percentage_24h')
            top_losers = df_filtered.nsmallest(5, 'price_change_percentage_24h')
            performance_df = pd.concat([top_gainers, top_losers])
            
            fig = px.bar(
                performance_df,
                x='price_change_percentage_24h',
                y='symbol',
                orientation='h',
                color='price_change_percentage_24h',
                color_continuous_scale='RdYlGn',
                title=""
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(26, 32, 44, 0.8)',
                paper_bgcolor='rgba(26, 32, 44, 0.8)',
                font={'color': '#f0f2f6'},
                xaxis_title="24h Change (%)",
                yaxis_title=""
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price change data not available")
    else:
        st.info("Insufficient data for performance chart")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Row 2: Risk Analysis
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.markdown("### ⚠️ Risk Assessment")

if len(df_filtered) > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk Score Distribution
        if 'risk_score' in df_filtered.columns:
            fig = px.histogram(
                df_filtered,
                x='risk_score',
                nbins=20,
                title="Risk Score Distribution"
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(26, 32, 44, 0.8)',
                paper_bgcolor='rgba(26, 32, 44, 0.8)',
                font={'color': '#f0f2f6'},
                xaxis_title="Risk Score",
                yaxis_title="Count"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk score not available")
    
    with col2:
        # High Risk Coins Table
        st.markdown("#### 🔴 High Risk Coins")
        
        if 'risk_score' in df_filtered.columns and 'symbol' in df_filtered.columns:
            high_risk = df_filtered.nlargest(5, 'risk_score')
            
            if len(high_risk) > 0:
                display_cols = ['symbol', 'risk_score']
                if 'price_change_percentage_24h' in high_risk.columns:
                    display_cols.append('price_change_percentage_24h')
                if 'volatility_24h' in high_risk.columns:
                    display_cols.append('volatility_24h')
                
                display_df = high_risk[display_cols].copy()
                if 'risk_score' in display_df.columns:
                    display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.1%}")
                if 'price_change_percentage_24h' in display_df.columns:
                    display_df['price_change_percentage_24h'] = display_df['price_change_percentage_24h'].apply(lambda x: f"{x:+.1f}%")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No high risk coins found")
        else:
            st.info("Risk data not available")

st.markdown('</div>', unsafe_allow_html=True)

# Data Table
st.markdown("---")
with st.expander("📋 View Filtered Data", expanded=False):
    st.markdown(f"### Filtered Data ({len(df_filtered)} rows)")
    
    # Select columns to display
    display_cols = ['name', 'symbol', 'market_cap_rank']
    if 'market_cap_formatted' in df_filtered.columns:
        display_cols.append('market_cap_formatted')
    if 'current_price' in df_filtered.columns:
        display_cols.append('current_price')
    if 'price_change_percentage_24h' in df_filtered.columns:
        display_cols.append('price_change_percentage_24h')
    if 'market_cap_category' in df_filtered.columns:
        display_cols.append('market_cap_category')
    if 'risk_score' in df_filtered.columns:
        display_cols.append('risk_score')
    
    display_df = df_filtered[display_cols].head(50).copy()
    
    # Format columns
    if 'current_price' in display_df.columns:
        display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
    
    if 'price_change_percentage_24h' in display_df.columns:
        display_df['price_change_percentage_24h'] = display_df['price_change_percentage_24h'].apply(lambda x: f"{x:+.1f}%")
    
    if 'risk_score' in display_df.columns:
        display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(display_df, use_container_width=True, height=400)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📅 Data updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
with col2:
    st.caption(f"🔍 Filters applied: Rank {rank_range[0]}-{rank_range[1]}")
with col3:
    st.caption("💡 Preprocessing: " + missing_method[:15] + "...")
