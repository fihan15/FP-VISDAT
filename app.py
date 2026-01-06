import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Crypto Market Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# =====================
# CUSTOM STYLE
# =====================
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0e1117; }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 { 
        color: white; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #9aa0a6;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1d29;
        padding: 8px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #303347;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e88e5;
        color: white;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background-color: #1a1d29;
    }
    
    /* Cards */
    .card {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 4px solid #1e88e5;
    }
    
    /* Success/Error colors */
    .positive { color: #00C853; }
    .negative { color: #FF5252; }
    .neutral { color: #FFC107; }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1d29;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1e88e5;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    """Load dataset dengan error handling"""
    try:
        df = pd.read_csv("crypto_top1000_dataset.csv")
        st.success(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        st.error("❌ File 'crypto_top1000_dataset.csv' not found. Please ensure the file exists.")
        # Create sample data for demonstration
        st.info("Creating sample data for demonstration...")
        np.random.seed(42)
        n = 1000
        data = {
            'id': range(1, n+1),
            'name': [f'Crypto{i}' for i in range(1, n+1)],
            'symbol': [f'CRYPTO{i:03d}' for i in range(1, n+1)],
            'market_cap_rank': range(1, n+1),
            'market_cap': np.random.lognormal(20, 2, n),
            'current_price': np.random.uniform(0.01, 50000, n),
            'total_volume': np.random.lognormal(15, 2, n),
            'high_24h': np.random.uniform(1, 60000, n),
            'low_24h': np.random.uniform(0.01, 40000, n),
            'price_change_percentage_24h': np.random.normal(0, 10, n),
            'price_change_percentage_7d': np.random.normal(2, 15, n),
            'price_change_percentage_30d': np.random.normal(5, 20, n),
            'price_change_percentage_1y': np.random.normal(20, 30, n),
            'circulating_supply': np.random.lognormal(15, 1.5, n),
            'total_supply': np.random.lognormal(16, 1.5, n),
            'max_supply': np.random.lognormal(17, 1.5, n),
            'fully_diluted_valuation': np.random.lognormal(21, 2, n),
            'ath': np.random.uniform(10, 100000, n),
            'ath_change_percentage': np.random.uniform(-90, 100, n),
        }
        df = pd.DataFrame(data)
        return df

df_raw = load_data()

# =====================
# DATA PROCESSING PIPELINE - COMPREHENSIVE
# =====================
st.sidebar.title("🔧 Data Processing Pipeline")

# 1. DATA PROCESSING SETTINGS
with st.sidebar.expander("⚙️ Processing Settings", expanded=True):
    processing_steps = st.multiselect(
        "Select Processing Steps:",
        ["Missing Value Handling", "Outlier Detection", "Feature Scaling", "Feature Engineering"],
        default=["Missing Value Handling", "Outlier Detection", "Feature Engineering"]
    )
    
    show_processing_stats = st.checkbox("Show Detailed Processing Stats", True)

# 2. MISSING VALUE HANDLING - ADVANCED
def handle_missing_values(df, method='advanced'):
    """Advanced missing value handling with multiple strategies"""
    
    df_missing = df.copy()
    stats = {}
    
    # Identify columns with missing values
    missing_cols = df_missing.columns[df_missing.isnull().any()].tolist()
    missing_counts = df_missing[missing_cols].isnull().sum()
    
    stats['before'] = {
        'total_missing': df_missing.isnull().sum().sum(),
        'missing_cols': len(missing_cols),
        'missing_percentage': (df_missing.isnull().sum().sum() / (df_missing.shape[0] * df_missing.shape[1])) * 100
    }
    
    # Strategy 1: Critical columns - drop rows if missing
    critical_cols = ['name', 'symbol', 'market_cap', 'current_price', 'market_cap_rank']
    critical_cols = [col for col in critical_cols if col in df_missing.columns]
    
    for col in critical_cols:
        if col in missing_cols:
            initial_rows = len(df_missing)
            df_missing = df_missing.dropna(subset=[col])
            stats[f'dropped_{col}'] = initial_rows - len(df_missing)
    
    # Strategy 2: Numeric columns - impute based on data distribution
    numeric_cols = df_missing.select_dtypes(include=[np.number]).columns.tolist()
    numeric_missing = [col for col in numeric_cols if col in missing_cols]
    
    if numeric_missing:
        # Group by market cap rank for smarter imputation
        df_missing['rank_group'] = pd.qcut(df_missing['market_cap_rank'], q=5, labels=False)
        
        for col in numeric_missing:
            if df_missing[col].isnull().sum() > 0:
                # Use group median for imputation
                df_missing[col] = df_missing.groupby('rank_group')[col].transform(
                    lambda x: x.fillna(x.median() if not x.median() != x.median() else 0)
                )
                # If still missing, use overall median
                df_missing[col] = df_missing[col].fillna(df_missing[col].median())
    
    # Strategy 3: Create missing indicators
    indicator_cols = ['max_supply', 'fully_diluted_valuation', 'price_change_percentage_1y', 'ath']
    for col in indicator_cols:
        if col in df_missing.columns:
            df_missing[f'has_{col}'] = df_missing[col].notna().astype(int)
    
    # Strategy 4: Fill specific columns with meaningful values
    fill_values = {
        'max_supply': -1,  # -1 indicates unlimited/infinite supply
        'fully_diluted_valuation': 0,
        'total_supply': df_missing['circulating_supply'] if 'circulating_supply' in df_missing.columns else np.nan,
    }
    
    for col, fill_val in fill_values.items():
        if col in df_missing.columns:
            df_missing[col] = df_missing[col].fillna(fill_val)
    
    # Clean up
    if 'rank_group' in df_missing.columns:
        df_missing = df_missing.drop('rank_group', axis=1)
    
    stats['after'] = {
        'total_missing': df_missing.isnull().sum().sum(),
        'missing_percentage': (df_missing.isnull().sum().sum() / (df_missing.shape[0] * df_missing.shape[1])) * 100,
        'rows_remaining': len(df_missing),
        'percent_rows_kept': (len(df_missing) / len(df)) * 100
    }
    
    return df_missing, stats

# 3. OUTLIER DETECTION & HANDLING - ADVANCED
def detect_and_handle_outliers(df, method='iqr', threshold=1.5):
    """Advanced outlier detection with multiple methods"""
    
    df_outlier = df.copy()
    outlier_stats = {}
    
    # Select numeric columns for outlier analysis
    numeric_cols = df_outlier.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove indicator columns and IDs
    exclude_cols = ['market_cap_rank', 'id', 'has_max_supply', 'has_fully_diluted_valuation', 
                   'has_price_change_percentage_1y', 'has_ath']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    outlier_stats['columns_analyzed'] = len(numeric_cols)
    outlier_stats['method'] = method
    outlier_stats['threshold'] = threshold
    
    for col in numeric_cols[:10]:  # Limit to first 10 columns for performance
        original_data = df_outlier[col].copy()
        outliers_mask = pd.Series([False] * len(df_outlier))
        
        if method == 'iqr':
            # IQR Method
            Q1 = df_outlier[col].quantile(0.25)
            Q3 = df_outlier[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers_mask = (df_outlier[col] < lower_bound) | (df_outlier[col] > upper_bound)
            
            # Winsorizing (cap outliers)
            df_outlier[col] = df_outlier[col].clip(lower_bound, upper_bound)
            
        elif method == 'zscore':
            # Z-Score Method
            mean = df_outlier[col].mean()
            std = df_outlier[col].std()
            z_scores = np.abs((df_outlier[col] - mean) / std)
            outliers_mask = z_scores > threshold
            
            # Cap at threshold * std
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            df_outlier[col] = df_outlier[col].clip(lower_bound, upper_bound)
            
        elif method == 'percentile':
            # Percentile Method
            lower_bound = df_outlier[col].quantile(0.01)
            upper_bound = df_outlier[col].quantile(0.99)
            outliers_mask = (df_outlier[col] < lower_bound) | (df_outlier[col] > upper_bound)
            
            df_outlier[col] = df_outlier[col].clip(lower_bound, upper_bound)
        
        # Calculate statistics
        outlier_count = outliers_mask.sum()
        outlier_percentage = (outlier_count / len(df_outlier)) * 100
        
        if outlier_count > 0:
            change_pct = ((df_outlier[col] - original_data).abs().sum() / original_data.abs().sum() * 100 
                         if original_data.abs().sum() > 0 else 0)
            
            outlier_stats[col] = {
                'outliers': outlier_count,
                'outlier_pct': outlier_percentage,
                'change_pct': change_pct,
                'original_mean': original_data.mean(),
                'processed_mean': df_outlier[col].mean()
            }
    
    outlier_stats['total_outliers_found'] = sum([stats.get('outliers', 0) for stats in outlier_stats.values() 
                                                if isinstance(stats, dict)])
    
    return df_outlier, outlier_stats

# 4. FEATURE SCALING - ADVANCED
def scale_features(df, method='robust', columns=None):
    """Advanced feature scaling with multiple methods"""
    
    df_scaled = df.copy()
    scaling_stats = {}
    
    # Select columns to scale
    if columns is None:
        # Default columns that benefit from scaling
        default_scale_cols = ['market_cap', 'total_volume', 'current_price', 'circulating_supply', 
                             'total_supply', 'fully_diluted_valuation', 'ath']
        scale_cols = [col for col in default_scale_cols if col in df_scaled.columns]
    else:
        scale_cols = [col for col in columns if col in df_scaled.columns]
    
    scaling_stats['columns_scaled'] = len(scale_cols)
    scaling_stats['method'] = method
    
    # Store original stats
    original_stats = {}
    for col in scale_cols:
        original_stats[col] = {
            'min': df_scaled[col].min(),
            'max': df_scaled[col].max(),
            'mean': df_scaled[col].mean(),
            'std': df_scaled[col].std()
        }
    
    # Apply scaling
    if method == 'robust':
        scaler = RobustScaler()
        df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
        
    elif method == 'minmax':
        scaler = MinMaxScaler()
        df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
        
    elif method == 'standard':
        scaler = StandardScaler()
        df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
        
    elif method == 'log':
        # Log transformation for highly skewed data
        for col in scale_cols:
            if df_scaled[col].min() > 0:
                df_scaled[col] = np.log1p(df_scaled[col])
            else:
                # Shift to positive then log
                min_val = df_scaled[col].min()
                if min_val <= 0:
                    shift = abs(min_val) + 1
                    df_scaled[col] = np.log1p(df_scaled[col] + shift)
    
    # Store processed stats
    processed_stats = {}
    for col in scale_cols:
        processed_stats[col] = {
            'min': df_scaled[col].min(),
            'max': df_scaled[col].max(),
            'mean': df_scaled[col].mean(),
            'std': df_scaled[col].std()
        }
    
    scaling_stats['original'] = original_stats
    scaling_stats['processed'] = processed_stats
    
    return df_scaled, scaling_stats

# 5. FEATURE ENGINEERING - COMPREHENSIVE
def engineer_features(df):
    """Create comprehensive features for crypto analysis"""
    
    df_features = df.copy()
    
    # 5.1 Volatility Features
    df_features["volatility_24h"] = (df_features["high_24h"] - df_features["low_24h"]) / df_features["current_price"]
    df_features["price_range_percentage"] = ((df_features["high_24h"] - df_features["low_24h"]) / df_features["low_24h"]) * 100
    
    # 5.2 Volume Analysis Features
    df_features["volume_marketcap_ratio"] = df_features["total_volume"] / df_features["market_cap"]
    df_features["volume_price_ratio"] = df_features["total_volume"] / df_features["current_price"]
    df_features["volume_velocity"] = df_features["volume_marketcap_ratio"] * df_features["price_change_percentage_24h"]
    
    # 5.3 Supply Analysis Features
    if 'circulating_supply' in df_features.columns and 'max_supply' in df_features.columns:
        df_features["supply_utilization"] = np.where(
            df_features["max_supply"] > 0,
            df_features["circulating_supply"] / df_features["max_supply"],
            1.0  # If max_supply is 0 or negative, assume 100% utilization
        )
        df_features["supply_inflation_risk"] = 1 - df_features["supply_utilization"].fillna(0)
        df_features["supply_growth_potential"] = df_features["max_supply"] - df_features["circulating_supply"]
    
    if 'total_supply' in df_features.columns:
        df_features["circulation_ratio"] = df_features["circulating_supply"] / df_features["total_supply"]
    
    # 5.4 Valuation Features
    if 'fully_diluted_valuation' in df_features.columns:
        df_features["fdv_mc_ratio"] = df_features["fully_diluted_valuation"] / df_features["market_cap"]
        df_features["fdv_mc_ratio"] = df_features["fdv_mc_ratio"].replace([np.inf, -np.inf], np.nan)
        df_features["valuation_gap"] = df_features["fdv_mc_ratio"] - 1
    
    # 5.5 Performance Features
    performance_cols = ['price_change_percentage_24h', 'price_change_percentage_7d', 
                       'price_change_percentage_30d', 'price_change_percentage_1y']
    
    available_perf_cols = [col for col in performance_cols if col in df_features.columns]
    
    if len(available_perf_cols) >= 2:
        # Short-term performance score
        if 'price_change_percentage_24h' in available_perf_cols and 'price_change_percentage_7d' in available_perf_cols:
            df_features["short_term_score"] = (
                df_features["price_change_percentage_24h"] * 0.6 +
                df_features["price_change_percentage_7d"] * 0.4
            )
        
        # Momentum indicator
        if 'price_change_percentage_24h' in available_perf_cols and 'price_change_percentage_7d' in available_perf_cols:
            df_features["momentum"] = df_features["price_change_percentage_24h"] - df_features["price_change_percentage_7d"]
    
    # 5.6 Risk Features
    df_features["risk_score"] = (
        df_features["volatility_24h"].rank(pct=True) * 0.4 +
        (df_features["price_change_percentage_24h"] < 0).astype(int) * 0.3 +
        df_features["supply_inflation_risk"].fillna(0).rank(pct=True) * 0.3
    )
    
    # Market cap based risk classification
    df_features["market_cap_category"] = pd.qcut(
        df_features["market_cap"], 
        q=4, 
        labels=["Micro Cap", "Small Cap", "Mid Cap", "Large Cap"]
    )
    
    # 5.7 Liquidity Features
    df_features["liquidity_score"] = (df_features["volume_marketcap_ratio"] * 100).rank(pct=True)
    
    # 5.8 Composite Score
    df_features["composite_score"] = (
        df_features["price_change_percentage_24h"].rank(pct=True) * 0.3 +
        (1 - df_features["risk_score"]) * 0.3 +  # Inverse of risk
        df_features["liquidity_score"] * 0.2 +
        df_features["supply_utilization"].fillna(0.5).rank(pct=True) * 0.2
    )
    
    # 5.9 Price Position Features
    if 'ath' in df_features.columns and 'ath_change_percentage' in df_features.columns:
        df_features["ath_distance"] = df_features["ath_change_percentage"] / 100
        df_features["recovery_potential"] = -df_features["ath_distance"]  # Positive value means room to recover
    
    return df_features

# =====================
# APPLY PROCESSING PIPELINE
# =====================
df_processed = df_raw.copy()
processing_summary = {}

# 1. Missing Value Handling
if "Missing Value Handling" in processing_steps:
    df_processed, missing_stats = handle_missing_values(df_processed)
    processing_summary['missing_values'] = missing_stats

# 2. Outlier Detection & Handling
if "Outlier Detection" in processing_steps:
    with st.sidebar.expander("Outlier Settings"):
        outlier_method = st.selectbox(
            "Outlier Detection Method",
            ["iqr", "zscore", "percentile"],
            index=0
        )
        outlier_threshold = st.slider("Outlier Threshold", 1.0, 3.0, 1.5, 0.1)
    
    df_processed, outlier_stats = detect_and_handle_outliers(
        df_processed, 
        method=outlier_method,
        threshold=outlier_threshold
    )
    processing_summary['outliers'] = outlier_stats

# 3. Feature Scaling
if "Feature Scaling" in processing_steps:
    with st.sidebar.expander("Scaling Settings"):
        scaling_method = st.selectbox(
            "Scaling Method",
            ["robust", "minmax", "standard", "log", "none"],
            index=0
        )
        
        if scaling_method != "none":
            default_cols = ['market_cap', 'total_volume', 'current_price', 'circulating_supply']
            scaling_cols = st.multiselect(
                "Columns to Scale",
                options=df_processed.select_dtypes(include=[np.number]).columns.tolist(),
                default=default_cols
            )
            
            df_processed, scaling_stats = scale_features(
                df_processed, 
                method=scaling_method,
                columns=scaling_cols
            )
            processing_summary['scaling'] = scaling_stats

# 4. Feature Engineering
if "Feature Engineering" in processing_steps:
    df_processed = engineer_features(df_processed)
    
    # Count new features
    original_cols = set(df_raw.columns)
    processed_cols = set(df_processed.columns)
    new_features = processed_cols - original_cols
    processing_summary['feature_engineering'] = {
        'new_features_count': len(new_features),
        'new_features': list(new_features)[:10]  # Show first 10
    }

# =====================
# DISPLAY PROCESSING SUMMARY
# =====================
if show_processing_stats:
    with st.expander("📊 Data Processing Summary", expanded=False):
        cols = st.columns(4)
        
        with cols[0]:
            st.metric("Original Data", 
                     f"{df_raw.shape[0]:,} rows\n{df_raw.shape[1]:,} cols",
                     delta=f"{df_processed.shape[1] - df_raw.shape[1]:+,} features")
        
        with cols[1]:
            if 'missing_values' in processing_summary:
                before = processing_summary['missing_values']['before']['total_missing']
                after = processing_summary['missing_values']['after']['total_missing']
                st.metric("Missing Values", f"{after:,}", 
                         delta=f"{-before + after:,}",
                         delta_color="inverse")
        
        with cols[2]:
            if 'outliers' in processing_summary:
                total_outliers = processing_summary['outliers']['total_outliers_found']
                st.metric("Outliers Handled", f"{total_outliers:,}")
        
        with cols[3]:
            if 'feature_engineering' in processing_summary:
                new_count = processing_summary['feature_engineering']['new_features_count']
                st.metric("New Features", new_count)
        
        # Show detailed stats in tabs
        detail_tabs = st.tabs(["Missing Values", "Outliers", "Scaling", "New Features"])
        
        with detail_tabs[0]:
            if 'missing_values' in processing_summary:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Before Processing:**")
                    st.json(processing_summary['missing_values']['before'])
                with col2:
                    st.write("**After Processing:**")
                    st.json(processing_summary['missing_values']['after'])
        
        with detail_tabs[1]:
            if 'outliers' in processing_summary:
                st.write(f"**Method:** {processing_summary['outliers']['method']}")
                st.write(f"**Threshold:** {processing_summary['outliers']['threshold']}")
                
                outlier_df = pd.DataFrame([
                    {**{'column': col}, **stats}
                    for col, stats in processing_summary['outliers'].items()
                    if isinstance(stats, dict) and 'outliers' in stats
                ])
                
                if not outlier_df.empty:
                    st.dataframe(outlier_df, use_container_width=True)
        
        with detail_tabs[2]:
            if 'scaling' in processing_summary:
                st.write(f"**Method:** {processing_summary['scaling']['method']}")
                st.write(f"**Columns Scaled:** {processing_summary['scaling']['columns_scaled']}")
                
                # Show before/after comparison for first 3 columns
                if 'original' in processing_summary['scaling']:
                    compare_cols = list(processing_summary['scaling']['original'].keys())[:3]
                    for col in compare_cols:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**{col} - Original:**")
                            st.json(processing_summary['scaling']['original'][col])
                        with col2:
                            st.write(f"**{col} - Scaled:**")
                            st.json(processing_summary['scaling']['processed'][col])
        
        with detail_tabs[3]:
            if 'feature_engineering' in processing_summary:
                st.write(f"**Total New Features:** {processing_summary['feature_engineering']['new_features_count']}")
                st.write("**New Features Created:**")
                
                # Group features by type
                volatility_features = [f for f in processing_summary['feature_engineering']['new_features'] 
                                     if 'volatility' in f.lower() or 'range' in f.lower()]
                volume_features = [f for f in processing_summary['feature_engineering']['new_features'] 
                                 if 'volume' in f.lower() or 'liquidity' in f.lower()]
                supply_features = [f for f in processing_summary['feature_engineering']['new_features'] 
                                 if 'supply' in f.lower()]
                risk_features = [f for f in processing_summary['feature_engineering']['new_features'] 
                               if 'risk' in f.lower() or 'score' in f.lower()]
                
                feature_groups = {
                    "📈 Volatility Features": volatility_features,
                    "💰 Volume & Liquidity": volume_features,
                    "🔄 Supply Analysis": supply_features,
                    "⚠️ Risk & Scoring": risk_features
                }
                
                for group_name, features in feature_groups.items():
                    if features:
                        st.write(f"**{group_name}**")
                        for feature in features:
                            st.write(f"- {feature}")

# =====================
# HELPER FUNCTIONS
# =====================
def categorize_market_cap(rank):
    """Categorize by market cap rank"""
    if rank <= 10: return "Mega Cap"
    if rank <= 50: return "Large Cap"
    if rank <= 200: return "Mid Cap"
    return "Small Cap"

def get_color_for_change(value):
    """Get color based on price change"""
    if value > 5: return '#00C853'  # Bright green
    elif value > 0: return '#4CAF50'  # Green
    elif value < -5: return '#FF5252'  # Bright red
    elif value < 0: return '#F44336'  # Red
    else: return '#FFC107'  # Yellow

# =====================
# SIDEBAR FILTERS
# =====================
st.sidebar.markdown("---")
st.sidebar.title("🎯 Dashboard Filters")

# Market Cap Rank Filter
rank_range = st.sidebar.slider(
    "Market Cap Rank Range",
    1, min(1000, len(df_processed)),
    (1, 100),
    help="Filter cryptocurrencies by market cap rank"
)

# Performance Filter
performance_filter = st.sidebar.selectbox(
    "Performance Filter",
    ["All", "Gainers Only (24h)", "Losers Only (24h)", "High Volatility", "High Volume"],
    help="Filter based on performance characteristics"
)

# Market Cap Category Filter
if 'market_cap_category' in df_processed.columns:
    categories = df_processed['market_cap_category'].unique()
    selected_categories = st.sidebar.multiselect(
        "Market Cap Categories",
        options=categories.tolist(),
        default=categories.tolist(),
        help="Filter by market capitalization size"
    )

# Risk Level Filter
if 'risk_score' in df_processed.columns:
    risk_level = st.sidebar.slider(
        "Maximum Risk Score",
        0.0, 1.0, 1.0, 0.1,
        help="Filter by maximum risk tolerance"
    )

# Apply filters
df_filtered = df_processed[
    (df_processed["market_cap_rank"] >= rank_range[0]) &
    (df_processed["market_cap_rank"] <= rank_range[1])
]

if performance_filter == "Gainers Only (24h)":
    df_filtered = df_filtered[df_filtered["price_change_percentage_24h"] > 0]
elif performance_filter == "Losers Only (24h)":
    df_filtered = df_filtered[df_filtered["price_change_percentage_24h"] < 0]
elif performance_filter == "High Volatility":
    if 'volatility_24h' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["volatility_24h"] > df_filtered["volatility_24h"].quantile(0.75)]
elif performance_filter == "High Volume":
    if 'volume_marketcap_ratio' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["volume_marketcap_ratio"] > df_filtered["volume_marketcap_ratio"].quantile(0.75)]

if 'market_cap_category' in df_processed.columns and 'selected_categories' in locals():
    df_filtered = df_filtered[df_filtered['market_cap_category'].isin(selected_categories)]

if 'risk_score' in df_processed.columns and 'risk_level' in locals():
    df_filtered = df_filtered[df_filtered['risk_score'] <= risk_level]

# Add market cap category for visualization
df_filtered["market_cap_group"] = df_filtered["market_cap_rank"].apply(categorize_market_cap)

# =====================
# MAIN DASHBOARD LAYOUT
# =====================
st.title("📊 Crypto Market Intelligence Dashboard")
st.markdown("""
<div style='background-color: #1a1d29; padding: 20px; border-radius: 10px; border-left: 4px solid #1e88e5;'>
    <p style='margin: 0; color: #9aa0a6;'>
    Advanced cryptocurrency market analysis with comprehensive data processing pipeline. 
    Track market trends, identify opportunities, and assess risks with interactive visualizations.
    </p>
</div>
""", unsafe_allow_html=True)

# =====================
# KEY METRICS
# =====================
st.markdown("### 📈 Market Overview")
metric_cols = st.columns(6)

with metric_cols[0]:
    st.metric(
        "Total Coins", 
        f"{len(df_filtered):,}",
        help="Number of cryptocurrencies in current filter"
    )

with metric_cols[1]:
    avg_change = df_filtered["price_change_percentage_24h"].mean()
    color = "normal"
    if avg_change > 0: color = "inverse"
    st.metric(
        "Avg 24h Change", 
        f"{avg_change:.2f}%",
        delta_color=color,
        help="Average 24-hour price change"
    )

with metric_cols[2]:
    if 'volatility_24h' in df_filtered.columns:
        avg_vol = df_filtered["volatility_24h"].mean() * 100
        st.metric(
            "Avg Volatility", 
            f"{avg_vol:.2f}%",
            help="Average 24-hour volatility"
        )

with metric_cols[3]:
    gainers = (df_filtered["price_change_percentage_24h"] > 0).sum()
    total = len(df_filtered)
    gainer_ratio = (gainers / total * 100) if total > 0 else 0
    st.metric(
        "Gainers (24h)", 
        f"{gainers}/{total}",
        f"{gainer_ratio:.1f}%",
        help="Number and percentage of cryptocurrencies with positive 24h change"
    )

with metric_cols[4]:
    total_market_cap = df_filtered["market_cap"].sum()
    st.metric(
        "Total Market Cap", 
        f"${total_market_cap:,.0f}",
        help="Sum of market capitalization for filtered cryptocurrencies"
    )

with metric_cols[5]:
    if 'risk_score' in df_filtered.columns:
        avg_risk = df_filtered["risk_score"].mean() * 100
        st.metric(
            "Avg Risk Score", 
            f"{avg_risk:.1f}/100",
            help="Average risk score (higher = more risky)"
        )

# =====================
# MAIN DASHBOARD TABS
# =====================
main_tabs = st.tabs(["📊 Market Analysis", "🏆 Top Performers", "📈 Technical Insights", "⚠️ Risk Assessment"])

# TAB 1: MARKET ANALYSIS
with main_tabs[0]:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🏆 Market Cap Distribution")
        
        # Top 10 by Market Cap
        top_10 = df_filtered.nsmallest(10, "market_cap_rank")
        
        fig = px.bar(
            top_10,
            x="symbol",
            y="market_cap",
            color="price_change_percentage_24h",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            text_auto=".2s",
            template="plotly_dark",
            title="Top 10 Cryptocurrencies by Market Cap"
        )
        fig.update_layout(
            xaxis_title="Cryptocurrency",
            yaxis_title="Market Cap",
            coloraxis_colorbar_title="24h Change (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Market Cap Category Distribution
        if 'market_cap_category' in df_filtered.columns:
            st.markdown("##### 📦 Market Cap Categories")
            category_counts = df_filtered['market_cap_category'].value_counts()
            
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Viridis,
                template="plotly_dark"
            )
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### 📊 Price Change Distribution")
        
        fig = px.histogram(
            df_filtered,
            x="price_change_percentage_24h",
            nbins=30,
            color_discrete_sequence=['#1e88e5'],
            template="plotly_dark",
            title="Distribution of 24-hour Price Changes"
        )
        fig.add_vline(
            x=0, 
            line_dash="dash", 
            line_color="white",
            annotation_text="Neutral",
            annotation_position="top right"
        )
        fig.update_layout(
            xaxis_title="24-hour Price Change (%)",
            yaxis_title="Number of Cryptocurrencies"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume vs Market Cap
        st.markdown("##### 💰 Volume/Market Cap Ratio")
        
        if 'volume_marketcap_ratio' in df_filtered.columns:
            fig = px.scatter(
                df_filtered.head(50),
                x="market_cap",
                y="volume_marketcap_ratio",
                color="price_change_percentage_24h",
                size="current_price",
                hover_name="name",
                color_continuous_scale="RdYlGn",
                template="plotly_dark",
                log_x=True,
                title="Liquidity Analysis"
            )
            fig.update_layout(
                xaxis_title="Market Cap (log scale)",
                yaxis_title="Volume/Market Cap Ratio",
                coloraxis_colorbar_title="24h Change (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

# TAB 2: TOP PERFORMERS
with main_tabs[1]:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📈 Top Gainers (24h)")
        
        top_gainers = df_filtered.nlargest(10, "price_change_percentage_24h")
        
        fig = px.bar(
            top_gainers,
            y="symbol",
            x="price_change_percentage_24h",
            orientation="h",
            color="price_change_percentage_24h",
            color_continuous_scale="Greens",
            text_auto=".1f",
            template="plotly_dark"
        )
        fig.update_layout(
            xaxis_title="24-hour Price Change (%)",
            yaxis_title="Cryptocurrency",
            coloraxis_colorbar_title="Change (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table
        st.markdown("**Detailed View:**")
        gainers_table = top_gainers[['symbol', 'name', 'price_change_percentage_24h', 
                                   'current_price', 'market_cap']].copy()
        gainers_table['price_change_percentage_24h'] = gainers_table['price_change_percentage_24h'].apply(
            lambda x: f"{x:+.2f}%")
        gainers_table['current_price'] = gainers_table['current_price'].apply(
            lambda x: f"${x:,.2f}")
        gainers_table['market_cap'] = gainers_table['market_cap'].apply(
            lambda x: f"${x:,.0f}")
        
        st.dataframe(gainers_table, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("##### 📉 Top Losers (24h)")
        
        top_losers = df_filtered.nsmallest(10, "price_change_percentage_24h")
        
        fig = px.bar(
            top_losers,
            y="symbol",
            x="price_change_percentage_24h",
            orientation="h",
            color="price_change_percentage_24h",
            color_continuous_scale="Reds_r",
            text_auto=".1f",
            template="plotly_dark"
        )
        fig.update_layout(
            xaxis_title="24-hour Price Change (%)",
            yaxis_title="Cryptocurrency",
            coloraxis_colorbar_title="Change (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table
        st.markdown("**Detailed View:**")
        losers_table = top_losers[['symbol', 'name', 'price_change_percentage_24h', 
                                 'current_price', 'market_cap']].copy()
        losers_table['price_change_percentage_24h'] = losers_table['price_change_percentage_24h'].apply(
            lambda x: f"{x:+.2f}%")
        losers_table['current_price'] = losers_table['current_price'].apply(
            lambda x: f"${x:,.2f}")
        losers_table['market_cap'] = losers_table['market_cap'].apply(
            lambda x: f"${x:,.0f}")
        
        st.dataframe(losers_table, use_container_width=True, hide_index=True)
    
    # Heatmap for correlation
    st.markdown("##### 🔗 Feature Correlation Heatmap")
    
    # Select numeric columns for correlation
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    # Select key columns for correlation
    key_cols = ['market_cap', 'current_price', 'total_volume', 'price_change_percentage_24h',
               'volatility_24h', 'volume_marketcap_ratio', 'risk_score', 'composite_score']
    available_key_cols = [col for col in key_cols if col in numeric_cols]
    
    if len(available_key_cols) > 1:
        corr_matrix = df_filtered[available_key_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            aspect="auto",
            template="plotly_dark",
            title="Correlation Matrix of Key Metrics"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: TECHNICAL INSIGHTS
with main_tabs[2]:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📊 Price vs Market Cap Analysis")
        
        # Ensure size values are positive
        if 'total_volume' in df_filtered.columns:
            df_filtered['size_normalized'] = (df_filtered['total_volume'] - df_filtered['total_volume'].min()) / \
                                           (df_filtered['total_volume'].max() - df_filtered['total_volume'].min()) * 50 + 10
        
        fig = px.scatter(
            df_filtered.head(100),
            x="current_price",
            y="market_cap",
            color="market_cap_group",
            size="size_normalized" if 'size_normalized' in df_filtered.columns else None,
            hover_name="name",
            hover_data=["price_change_percentage_24h", "volatility_24h", "volume_marketcap_ratio"],
            log_x=True,
            log_y=True,
            template="plotly_dark",
            title="Price vs Market Cap (Bubble size = Trading Volume)"
        )
        fig.update_layout(
            xaxis_title="Current Price (log scale)",
            yaxis_title="Market Cap (log scale)",
            legend_title="Market Cap Group"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Volatility Analysis
        st.markdown("##### 📈 Volatility Distribution")
        
        if 'volatility_24h' in df_filtered.columns:
            fig = px.box(
                df_filtered,
                x="market_cap_group",
                y="volatility_24h",
                color="market_cap_group",
                template="plotly_dark",
                title="Volatility by Market Cap Group"
            )
            fig.update_layout(
                xaxis_title="Market Cap Group",
                yaxis_title="24-hour Volatility",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### 🔄 Supply Analysis")
        
        if 'supply_utilization' in df_filtered.columns:
            fig = px.scatter(
                df_filtered,
                x="supply_utilization",
                y="price_change_percentage_24h",
                color="market_cap_group",
                size="market_cap",
                hover_name="name",
                template="plotly_dark",
                title="Supply Utilization vs Price Performance"
            )
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="white"
            )
            fig.update_layout(
                xaxis_title="Supply Utilization (%)",
                yaxis_title="24-hour Price Change (%)",
                legend_title="Market Cap Group"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # FDV/MC Ratio Analysis
        st.markdown("##### ⚖️ FDV/Market Cap Ratio")
        
        if 'fdv_mc_ratio' in df_filtered.columns:
            # Filter out infinite values
            fdv_data = df_filtered[df_filtered['fdv_mc_ratio'].between(0, 10)]
            
            if not fdv_data.empty:
                fig = px.histogram(
                    fdv_data,
                    x="fdv_mc_ratio",
                    nbins=30,
                    color_discrete_sequence=['#FF9800'],
                    template="plotly_dark",
                    title="Distribution of FDV/Market Cap Ratio"
                )
                fig.add_vline(
                    x=1,
                    line_dash="dash",
                    line_color="white",
                    annotation_text="FDV = Market Cap",
                    annotation_position="top right"
                )
                fig.update_layout(
                    xaxis_title="FDV / Market Cap Ratio",
                    yaxis_title="Number of Cryptocurrencies"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Time-based performance (if available)
    st.markdown("##### ⏰ Multi-timeframe Performance")
    
    time_cols = ['price_change_percentage_24h', 'price_change_percentage_7d', 
                'price_change_percentage_30d']
    available_time_cols = [col for col in time_cols if col in df_filtered.columns]
    
    if len(available_time_cols) > 1:
        # Select top 20 by market cap
        top_20 = df_filtered.nsmallest(20, "market_cap_rank")
        
        # Melt for grouped bar chart
        melted_data = top_20.melt(
            id_vars=['symbol'],
            value_vars=available_time_cols,
            var_name='timeframe',
            value_name='price_change'
        )
        
        # Clean up timeframe names
        timeframe_names = {
            'price_change_percentage_24h': '24h',
            'price_change_percentage_7d': '7d',
            'price_change_percentage_30d': '30d'
        }
        melted_data['timeframe'] = melted_data['timeframe'].map(timeframe_names)
        
        fig = px.bar(
            melted_data,
            x="symbol",
            y="price_change",
            color="timeframe",
            barmode="group",
            template="plotly_dark",
            title="Performance Across Different Timeframes (Top 20 by Market Cap)"
        )
        fig.update_layout(
            xaxis_title="Cryptocurrency",
            yaxis_title="Price Change (%)",
            legend_title="Timeframe"
        )
        st.plotly_chart(fig, use_container_width=True)

# TAB 4: RISK ASSESSMENT
with main_tabs[3]:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### ⚠️ Risk Score Distribution")
        
        if 'risk_score' in df_filtered.columns:
            # Create risk categories
            df_filtered['risk_category'] = pd.cut(
                df_filtered['risk_score'],
                bins=[0, 0.3, 0.7, 1],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            
            fig = px.pie(
                df_filtered,
                names='risk_category',
                hole=0.4,
                color='risk_category',
                color_discrete_map={
                    'Low Risk': '#00C853',
                    'Medium Risk': '#FFC107',
                    'High Risk': '#FF5252'
                },
                template="plotly_dark"
            )
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk vs Return
            st.markdown("##### 📊 Risk vs Return Analysis")
            
            fig = px.scatter(
                df_filtered,
                x="risk_score",
                y="price_change_percentage_24h",
                color="market_cap_group",
                size="market_cap",
                hover_name="name",
                template="plotly_dark",
                title="Risk-Return Profile"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="white")
            fig.add_vline(x=0.5, line_dash="dash", line_color="white")
            fig.update_layout(
                xaxis_title="Risk Score",
                yaxis_title="24-hour Return (%)",
                legend_title="Market Cap Group"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### 🚨 High Risk Cryptocurrencies")
        
        if 'risk_score' in df_filtered.columns:
            high_risk = df_filtered[df_filtered['risk_score'] > 0.7].nsmallest(10, "market_cap_rank")
            
            if not high_risk.empty:
                fig = px.bar(
                    high_risk,
                    y="symbol",
                    x="risk_score",
                    orientation="h",
                    color="volatility_24h" if 'volatility_24h' in high_risk.columns else "risk_score",
                    color_continuous_scale="Reds",
                    template="plotly_dark",
                    title="Top 10 High Risk Cryptocurrencies"
                )
                fig.update_layout(
                    xaxis_title="Risk Score",
                    yaxis_title="Cryptocurrency"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display high risk table
                st.markdown("**High Risk Details:**")
                risk_table = high_risk[['symbol', 'name', 'risk_score', 'volatility_24h', 
                                      'price_change_percentage_24h', 'market_cap']].copy()
                risk_table['risk_score'] = risk_table['risk_score'].apply(lambda x: f"{x:.3f}")
                risk_table['volatility_24h'] = risk_table['volatility_24h'].apply(
                    lambda x: f"{x*100:.2f}%")
                risk_table['price_change_percentage_24h'] = risk_table['price_change_percentage_24h'].apply(
                    lambda x: f"{x:+.2f}%")
                risk_table['market_cap'] = risk_table['market_cap'].apply(
                    lambda x: f"${x:,.0f}")
                
                st.dataframe(risk_table, use_container_width=True, hide_index=True)
            else:
                st.info("No high risk cryptocurrencies found in current filter.")
        
        # Composite Score Analysis
        st.markdown("##### 🏆 Composite Score Ranking")
        
        if 'composite_score' in df_filtered.columns:
            top_composite = df_filtered.nlargest(10, "composite_score")
            
            fig = px.bar(
                top_composite,
                y="symbol",
                x="composite_score",
                orientation="h",
                color="composite_score",
                color_continuous_scale="Viridis",
                template="plotly_dark",
                title="Top 10 by Composite Score"
            )
            fig.update_layout(
                xaxis_title="Composite Score",
                yaxis_title="Cryptocurrency"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk Radar Chart for top cryptocurrencies
    st.markdown("##### 📡 Multi-dimensional Risk Analysis")
    
    if len(df_filtered) >= 5:
        # Select top 5 by market cap
        top_5 = df_filtered.nsmallest(5, "market_cap_rank")
        
        # Select risk metrics
        risk_metrics = ['risk_score', 'volatility_24h', 'supply_inflation_risk', 
                       'volume_marketcap_ratio']
        available_risk_metrics = [m for m in risk_metrics if m in top_5.columns]
        
        if len(available_risk_metrics) >= 3:
            # Normalize metrics for radar chart
            radar_data = []
            
            for metric in available_risk_metrics:
                min_val = top_5[metric].min()
                max_val = top_5[metric].max()
                if max_val > min_val:
                    normalized = (top_5[metric] - min_val) / (max_val - min_val)
                else:
                    normalized = top_5[metric] * 0
                
                for idx, (_, row) in enumerate(top_5.iterrows()):
                    radar_data.append({
                        'symbol': row['symbol'],
                        'metric': metric.replace('_', ' ').title(),
                        'value': normalized.iloc[idx]
                    })
            
            radar_df = pd.DataFrame(radar_data)
            
            if not radar_df.empty:
                fig = px.line_polar(
                    radar_df,
                    r='value',
                    theta='metric',
                    color='symbol',
                    line_close=True,
                    template="plotly_dark"
                )
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=True,
                    title="Risk Profile Comparison (Normalized)"
                )
                st.plotly_chart(fig, use_container_width=True)

# =====================
# DATA DOWNLOAD
# =====================
st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Export Data")

# Convert filtered dataframe to CSV
csv = df_filtered.to_csv(index=False)
st.sidebar.download_button(
    label="📥 Download Filtered Data (CSV)",
    data=csv,
    file_name=f"crypto_filtered_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
    help="Download the currently filtered dataset as CSV"
)

# =====================
# FOOTER
# =====================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    **📊 Crypto Market Dashboard**  
    *Advanced Analytics Platform*
    """)

with footer_col2:
    st.markdown(f"""
    **🔄 Data Status**  
    Processed: {len(df_processed):,} records  
    Filtered: {len(df_filtered):,} records
    """)

with footer_col3:
    st.markdown(f"""
    **⏰ Last Updated**  
    {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)

st.caption("© 2024 Crypto Analytics Dashboard | Built with Streamlit & Plotly")
