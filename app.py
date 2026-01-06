import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =====================
# PAGE CONFIG & THEME - DARK MODE FRIENDLY
# =====================
st.set_page_config(
    page_title="📊 Crypto Market Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="📈"
)

# =====================
# CUSTOM STYLING - DARK MODE COMPATIBLE
# =====================
st.markdown("""
<style>
    /* Reset untuk dark mode compatibility */
    .main {
        background-color: #0e1117;
    }
    
    /* Dark mode text colors */
    .stApp {
        background-color: #0e1117;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #f0f2f6 !important;
    }
    
    /* Title styling */
    .main-title {
        font-size: 3.5rem !important;
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
        font-size: 1.2rem !important;
        color: #a0aec0 !important;
        text-align: center;
        margin-bottom: 2rem !important;
        font-weight: 300;
    }
    
    /* Card styling untuk metrics - Dark mode */
    .metric-card {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border-left: 5px solid #00d4ff;
        transition: transform 0.3s ease;
        margin-bottom: 15px;
        border: 1px solid #2d3748;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 212, 255, 0.1);
    }
    
    /* Chart container styling - Dark mode */
    .chart-container {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 25px;
        border: 1px solid #2d3748;
    }
    
    /* Section headers - Dark mode */
    .section-header {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #00d4ff !important;
        margin-top: 30px !important;
        margin-bottom: 20px !important;
        padding-bottom: 10px;
        border-bottom: 3px solid #00d4ff;
        display: inline-block;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Tab styling - Dark mode */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: #1a202c;
        border-radius: 10px 10px 0 0;
        padding: 10px 25px;
        font-weight: 500;
        color: #a0aec0;
        border: 1px solid #2d3748;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d4ff, #0083ff) !important;
        color: white !important;
        border-color: #00d4ff !important;
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.2);
    }
    
    /* Button styling - Dark mode */
    .stButton button {
        background: linear-gradient(90deg, #00d4ff, #0083ff);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 25px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 212, 255, 0.3);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom info boxes - Dark mode */
    .info-box {
        background: linear-gradient(135deg, #1a3c2e 0%, #2d5c4a 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00ff88;
        border: 1px solid #2d5c4a;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #3c2e1a 0%, #5c4a2d 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #ffaa00;
        border: 1px solid #5c4a2d;
    }
    
    /* Metric labels dan values - Dark mode */
    .stMetric {
        background: transparent !important;
    }
    
    .stMetric label {
        color: #a0aec0 !important;
        font-size: 14px !important;
    }
    
    .stMetric div[data-testid="stMetricValue"] {
        color: #f0f2f6 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    .stMetric div[data-testid="stMetricDelta"] {
        font-size: 14px !important;
    }
    
    /* Caption text - Dark mode */
    .caption-text {
        color: #718096 !important;
        font-size: 12px !important;
    }
    
    /* Dataframe styling - Dark mode */
    .dataframe {
        background: #1a202c !important;
        color: #f0f2f6 !important;
    }
    
    .dataframe th {
        background: #2d3748 !important;
        color: #00d4ff !important;
    }
    
    .dataframe td {
        background: #1a202c !important;
        color: #f0f2f6 !important;
        border-color: #2d3748 !important;
    }
    
    /* Expander styling - Dark mode */
    .streamlit-expanderHeader {
        background: #1a202c !important;
        color: #f0f2f6 !important;
        border: 1px solid #2d3748 !important;
    }
    
    /* Progress bar - Dark mode */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #0083ff) !important;
    }
    
    /* Slider - Dark mode */
    .stSlider > div > div > div {
        background: #00d4ff !important;
    }
    
    /* Selectbox - Dark mode */
    .stSelectbox > div > div {
        background: #1a202c !important;
        color: #f0f2f6 !important;
        border: 1px solid #2d3748 !important;
    }
    
    /* Plotly chart background fix */
    .js-plotly-plot {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# HELPER FUNCTIONS FOR DARK MODE PLOTS
# =====================
def get_dark_template():
    """Return plotly dark template configuration"""
    return {
        'layout': {
            'plot_bgcolor': 'rgba(26, 32, 44, 0.8)',
            'paper_bgcolor': 'rgba(26, 32, 44, 0.8)',
            'font': {'color': '#f0f2f6', 'family': 'Arial'},
            'title': {'font': {'color': '#00d4ff'}},
            'xaxis': {
                'gridcolor': '#2d3748',
                'linecolor': '#2d3748',
                'zerolinecolor': '#2d3748',
                'tickfont': {'color': '#a0aec0'}
            },
            'yaxis': {
                'gridcolor': '#2d3748',
                'linecolor': '#2d3748',
                'zerolinecolor': '#2d3748',
                'tickfont': {'color': '#a0aec0'}
            },
            'colorway': ['#00d4ff', '#0083ff', '#00ff88', '#ffaa00', '#ff5555'],
        }
    }

def create_dark_figure():
    """Create a figure with dark mode settings"""
    fig = go.Figure()
    fig.update_layout(**get_dark_template()['layout'])
    return fig

# =====================
# LOAD DATA WITH SAMPLE FOR PRESENTATION
# =====================
@st.cache_data
def load_sample_data():
    """Load data dengan sample untuk presentasi"""
    try:
        df = pd.read_csv("crypto_top1000_dataset.csv")
        st.success("✅ Real data loaded successfully")
    except:
        # Create sample data for presentation if file not found
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
        df['volatility_24h'] = (df['high_24h'] - df['low_24h']) / df['current_price']
        df['volume_marketcap_ratio'] = df['total_volume'] / df['market_cap']
        st.info("📊 Sample data generated for presentation")
    
    return df

df = load_sample_data()

# =====================
# DATA PROCESSING
# =====================
def process_data_for_presentation(df):
    """Simple data processing"""
    
    df_clean = df.copy()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Calculate essential metrics
    df_clean['volatility_24h'] = (df_clean['high_24h'] - df_clean['low_24h']) / df_clean['current_price']
    df_clean['volume_marketcap_ratio'] = df_clean['total_volume'] / df_clean['market_cap']
    
    # Create market cap categories
    df_clean['market_cap_category'] = pd.qcut(
        df_clean['market_cap'], 
        q=[0, 0.1, 0.5, 1], 
        labels=['Small Cap', 'Mid Cap', 'Large Cap']
    )
    
    # Format for display
    df_clean['market_cap_formatted'] = df_clean['market_cap'].apply(
        lambda x: f'${x/1e9:.1f}B' if x >= 1e9 else f'${x/1e6:.0f}M'
    )
    
    # Calculate scores
    df_clean['performance_score'] = (
        df_clean['price_change_percentage_24h'].rank(pct=True) * 0.7 -
        df_clean['volatility_24h'].rank(pct=True) * 0.3
    )
    
    df_clean['risk_score'] = (
        df_clean['volatility_24h'].rank(pct=True) * 0.4 +
        (df_clean['price_change_percentage_24h'] < 0).astype(float) * 0.3 +
        (1 - df_clean['market_cap'].rank(pct=True)) * 0.3
    )
    
    return df_clean

df_processed = process_data_for_presentation(df)

# =====================
# HERO SECTION - TITLE & KEY METRICS
# =====================
st.markdown('<h1 class="main-title">📈 Crypto Market Intelligence Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time Analysis & Insights for Informed Decision Making</p>', unsafe_allow_html=True)

# Create a top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_market_cap = df_processed['market_cap'].sum()
        st.metric(
            label="💰 **Total Market Cap**",
            value=f"${total_market_cap/1e12:.1f}T",
            delta="+2.3%",
            delta_color="normal"
        )
        st.markdown('<p class="caption-text">Total cryptocurrency market valuation</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_change = df_processed['price_change_percentage_24h'].mean()
        st.metric(
            label="📊 **24h Avg Change**",
            value=f"{avg_change:+.1f}%",
            delta_color="inverse" if avg_change < 0 else "normal"
        )
        st.markdown('<p class="caption-text">Average daily price movement</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        gainers = (df_processed['price_change_percentage_24h'] > 0).sum()
        total_coins = len(df_processed)
        st.metric(
            label="📈 **Market Sentiment**",
            value=f"{(gainers/total_coins*100):.0f}%",
            delta="Bullish" if gainers/total_coins > 0.5 else "Bearish"
        )
        st.markdown(f'<p class="caption-text">{gainers}/{total_coins} coins positive</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col4:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        top_10 = df_processed.nsmallest(10, 'market_cap_rank')
        top_10_dominance = top_10['market_cap'].sum() / total_market_cap * 100
        st.metric(
            label="👑 **Top 10 Dominance**",
            value=f"{top_10_dominance:.1f}%",
            delta="-1.2%"
        )
        st.markdown('<p class="caption-text">Market share of top 10 cryptocurrencies</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# =====================
# MARKET OVERVIEW SECTION
# =====================
st.markdown('<h2 class="section-header">📊 Market Overview</h2>', unsafe_allow_html=True)

# Row 1: Market Distribution & Performance
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 🗺️ Market Cap Distribution")
    
    # Treemap dengan dark mode
    fig = px.treemap(
        df_processed.head(30),
        path=['market_cap_category', 'symbol'],
        values='market_cap',
        color='price_change_percentage_24h',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        hover_data={'market_cap': ':.2s', 'current_price': ':.2f'},
    )
    
    # Apply dark theme
    fig.update_layout(
        **get_dark_template()['layout'],
        margin=dict(t=0, b=0, l=0, r=0)
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Market Health Indicators")
    
    # Buat gauge charts dengan dark mode
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}]],
        vertical_spacing=0.2,
        horizontal_spacing=0.2
    )
    
    # Gauge 1: Market Sentiment
    sentiment = (df_processed['price_change_percentage_24h'] > 0).mean() * 100
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=sentiment,
        domain={'x': [0, 0.45], 'y': [0.5, 1]},
        title={'text': "Bullish %", 'font': {'color': '#00d4ff'}},
        number={'font': {'color': '#f0f2f6'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#a0aec0'},
            'bar': {'color': "#00d4ff"},
            'bgcolor': '#1a202c',
            'borderwidth': 2,
            'bordercolor': '#2d3748',
            'steps': [
                {'range': [0, 30], 'color': "#4a1c1c"},
                {'range': [30, 70], 'color': "#4a3c1c"},
                {'range': [70, 100], 'color': "#1c4a2d"}
            ]
        }
    ), row=1, col=1)
    
    # Gauge 2: Volatility
    avg_vol = df_processed['volatility_24h'].mean() * 100
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=min(avg_vol, 20),
        domain={'x': [0.55, 1], 'y': [0.5, 1]},
        title={'text': "Avg Volatility", 'font': {'color': '#00d4ff'}},
        number={'font': {'color': '#f0f2f6'}},
        gauge={
            'axis': {'range': [0, 20], 'tickcolor': '#a0aec0'},
            'bar': {'color': "#ff5555"},
            'bgcolor': '#1a202c',
            'borderwidth': 2,
            'bordercolor': '#2d3748',
            'steps': [
                {'range': [0, 5], 'color': "#1c4a2d"},
                {'range': [5, 15], 'color': "#4a3c1c"},
                {'range': [15, 20], 'color': "#4a1c1c"}
            ]
        }
    ), row=1, col=2)
    
    # Gauge 3: Liquidity
    avg_volume_ratio = df_processed['volume_marketcap_ratio'].mean() * 100
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=min(avg_volume_ratio * 10, 10),
        domain={'x': [0, 0.45], 'y': [0, 0.5]},
        title={'text': "Liquidity", 'font': {'color': '#00d4ff'}},
        number={'font': {'color': '#f0f2f6'}},
        gauge={
            'axis': {'range': [0, 10], 'tickcolor': '#a0aec0'},
            'bar': {'color': "#0083ff"},
            'bgcolor': '#1a202c',
            'borderwidth': 2,
            'bordercolor': '#2d3748',
            'steps': [
                {'range': [0, 3], 'color': "#4a1c1c"},
                {'range': [3, 7], 'color': "#4a3c1c"},
                {'range': [7, 10], 'color': "#1c4a2d"}
            ]
        }
    ), row=2, col=1)
    
    # Gauge 4: Concentration Risk
    concentration = top_10_dominance
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=concentration,
        domain={'x': [0.55, 1], 'y': [0, 0.5]},
        title={'text': "Top 10 Share", 'font': {'color': '#00d4ff'}},
        number={'font': {'color': '#f0f2f6'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#a0aec0'},
            'bar': {'color': "#aa00ff"},
            'bgcolor': '#1a202c',
            'borderwidth': 2,
            'bordercolor': '#2d3748',
            'steps': [
                {'range': [0, 40], 'color': "#1c4a2d"},
                {'range': [40, 70], 'color': "#4a3c1c"},
                {'range': [70, 100], 'color': "#4a1c1c"}
            ]
        }
    ), row=2, col=2)
    
    fig.update_layout(
        height=400,
        plot_bgcolor='rgba(26, 32, 44, 0.8)',
        paper_bgcolor='rgba(26, 32, 44, 0.8)',
        font={'color': '#f0f2f6'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Row 2: Top Performers
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.markdown("### 🏆 Top Performers Analysis")

col1, col2 = st.columns(2)

with col1:
    # Top 5 Gainers
    top_gainers = df_processed.nlargest(5, 'price_change_percentage_24h')
    fig = px.bar(
        top_gainers,
        x='symbol',
        y='price_change_percentage_24h',
        color='price_change_percentage_24h',
        color_continuous_scale='Greens',
        text_auto='+.1f'
    )
    
    fig.update_layout(
        **get_dark_template()['layout'],
        showlegend=False,
        yaxis_title="Price Change %",
        xaxis_title="",
        title="🚀 Top 5 Gainers (24h)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Top 5 Losers
    top_losers = df_processed.nsmallest(5, 'price_change_percentage_24h')
    fig = px.bar(
        top_losers,
        x='symbol',
        y='price_change_percentage_24h',
        color='price_change_percentage_24h',
        color_continuous_scale='Reds',
        text_auto='+.1f'
    )
    
    fig.update_layout(
        **get_dark_template()['layout'],
        showlegend=False,
        yaxis_title="Price Change %",
        xaxis_title="",
        title="📉 Top 5 Losers (24h)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# =====================
# INSIGHTS & ANALYSIS SECTION
# =====================
st.markdown('<h2 class="section-header">🔍 Key Insights & Analysis</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📈 Price vs Market Cap Correlation")
    
    fig = px.scatter(
        df_processed.head(50),
        x='current_price',
        y='market_cap',
        color='market_cap_category',
        size='total_volume',
        hover_name='symbol',
        log_x=True,
        log_y=True,
        size_max=40,
        opacity=0.7,
        color_discrete_map={
            'Large Cap': '#00d4ff',
            'Mid Cap': '#ffaa00',
            'Small Cap': '#00ff88'
        }
    )
    
    fig.update_layout(
        **get_dark_template()['layout'],
        xaxis_title="Price (Log Scale)",
        yaxis_title="Market Cap (Log Scale)",
        legend_title="Market Cap Size"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Market Distribution Analysis")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("📦 Market Cap Distribution", "📈 Daily Returns Distribution"),
        vertical_spacing=0.15
    )
    
    # Histogram 1: Market Cap Distribution
    fig.add_trace(
        go.Histogram(
            x=np.log10(df_processed['market_cap']),
            nbinsx=30,
            marker_color='#00d4ff',
            opacity=0.7,
            name="Market Cap (Log10)"
        ),
        row=1, col=1
    )
    
    # Histogram 2: Returns Distribution
    fig.add_trace(
        go.Histogram(
            x=df_processed['price_change_percentage_24h'],
            nbinsx=30,
            marker_color='#0083ff',
            opacity=0.7,
            name="24h Returns"
        ),
        row=2, col=1
    )
    
    # Update layout with dark theme
    dark_layout = get_dark_template()['layout']
    fig.update_layout(
        height=600,
        **dark_layout,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Market Cap (Log10 $)", row=1, col=1)
    fig.update_xaxes(title_text="24h Price Change %", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    # Add mean lines
    fig.add_vline(
        x=np.log10(df_processed['market_cap'].mean()),
        row=1, col=1,
        line_dash="dash",
        line_color="#ff5555",
        annotation_text=f"Mean: ${df_processed['market_cap'].mean():.2e}"
    )
    
    fig.add_vline(
        x=df_processed['price_change_percentage_24h'].mean(),
        row=2, col=1,
        line_dash="dash",
        line_color="#ff5555",
        annotation_text=f"Mean: {df_processed['price_change_percentage_24h'].mean():.1f}%"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================
# RISK ASSESSMENT SECTION
# =====================
st.markdown('<h2 class="section-header">⚠️ Risk Assessment & Opportunities</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### ⚠️ Market Risk Indicators")
    
    # Hitung risk metrics
    if len(df_processed) > 0:
        # Volatility Risk
        volatility_raw = df_processed['volatility_24h'].abs().mean()
        volatility_risk = min(volatility_raw * 500, 100)
        
        # Market Sentiment Risk
        negative_coins = (df_processed['price_change_percentage_24h'] < 0).sum()
        sentiment_risk = (negative_coins / len(df_processed)) * 100
        
        # Concentration Risk
        top_5 = df_processed.nsmallest(5, 'market_cap_rank')
        concentration_risk = (top_5['market_cap'].sum() / total_market_cap) * 100
        
        # Composite Risk Score
        composite_risk = (
            volatility_risk * 0.4 +
            sentiment_risk * 0.3 +
            concentration_risk * 0.3
        )
        composite_risk = min(max(composite_risk, 0), 100)
        
        # Determine risk level
        if composite_risk < 30:
            risk_level = "🟢 LOW"
            risk_color = "#00ff88"
        elif composite_risk < 70:
            risk_level = "🟡 MEDIUM"
            risk_color = "#ffaa00"
        else:
            risk_level = "🔴 HIGH"
            risk_color = "#ff5555"
        
        # Buat gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=composite_risk,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Market Risk Score: {risk_level}", 'font': {'size': 24, 'color': risk_color}},
            number={'suffix': "/100", 'font': {'size': 40, 'color': risk_color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#a0aec0"},
                'bar': {'color': risk_color, 'thickness': 0.5},
                'bgcolor': "#1a202c",
                'borderwidth': 2,
                'bordercolor': "#2d3748",
                'steps': [
                    {'range': [0, 30], 'color': "#1c4a2d"},
                    {'range': [30, 70], 'color': "#4a3c1c"},
                    {'range': [70, 100], 'color': "#4a1c1c"}
                ],
                'threshold': {
                    'line': {'color': risk_color, 'width': 4},
                    'thickness': 0.75,
                    'value': composite_risk
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(26, 32, 44, 0.8)',
            paper_bgcolor='rgba(26, 32, 44, 0.8)',
            font={'color': "#f0f2f6", 'family': "Arial"},
            margin=dict(t=80, b=40, l=40, r=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display risk breakdown
        st.markdown("#### 🔍 Risk Breakdown Analysis")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            vol_level = "High" if volatility_risk > 60 else "Medium" if volatility_risk > 30 else "Low"
            st.metric(
                label="📈 **Volatility Risk**",
                value=f"{volatility_risk:.1f}/100",
                delta=vol_level
            )
            st.markdown(f'<p class="caption-text">Avg volatility: {(volatility_raw*100):.2f}%</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_b:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="📊 **Market Sentiment**",
                value=f"{sentiment_risk:.1f}/100",
                delta=f"{negative_coins}/{len(df_processed)} coins down"
            )
            st.markdown(f'<p class="caption-text">{(100-sentiment_risk):.1f}% of coins rising</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_c:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            conc_level = "High" if concentration_risk > 60 else "Medium" if concentration_risk > 30 else "Low"
            st.metric(
                label="👑 **Market Concentration**",
                value=f"{concentration_risk:.1f}/100",
                delta=conc_level
            )
            st.markdown(f'<p class="caption-text">Top 5 control {concentration_risk:.1f}% of market</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.info("No data available for risk assessment")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Risk by Market Cap Category")
    
    if len(df_processed) > 0 and 'market_cap_category' in df_processed.columns:
        # Hitung metrics per kategori
        category_stats = df_processed.groupby('market_cap_category').agg({
            'price_change_percentage_24h': ['mean', 'count'],
            'risk_score': 'mean'
        }).round(2)
        
        category_stats.columns = ['avg_return', 'count', 'avg_risk_score']
        category_stats = category_stats.reset_index()
        
        # Buat bar chart
        fig = px.bar(
            category_stats,
            x='market_cap_category',
            y='avg_return',
            color='market_cap_category',
            color_discrete_map={
                'Large Cap': '#00d4ff',
                'Mid Cap': '#ffaa00',
                'Small Cap': '#ff5555'
            },
            text='avg_return'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            **get_dark_template()['layout'],
            showlegend=False,
            height=300,
            xaxis_title="",
            yaxis_title="Avg Return %",
            title="Average 24h Returns by Category"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display insights
        st.markdown("#### 💡 Category Insights:")
        
        for category in ['Large Cap', 'Mid Cap', 'Small Cap']:
            if category in category_stats['market_cap_category'].values:
                cat_data = category_stats[category_stats['market_cap_category'] == category].iloc[0]
                
                if category == 'Large Cap':
                    st.markdown(f"""
                    <div style="background: #1c4a2d; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #00ff88;">
                        <strong style="color: #00ff88;">{category}</strong> ({int(cat_data['count'])} coins)
                        <br>
                        📈 Avg Return: {cat_data['avg_return']:+.1f}%
                        <br>
                        ⚠️ Risk Score: {cat_data['avg_risk_score']:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                elif category == 'Mid Cap':
                    st.markdown(f"""
                    <div style="background: #4a3c1c; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #ffaa00;">
                        <strong style="color: #ffaa00;">{category}</strong> ({int(cat_data['count'])} coins)
                        <br>
                        📈 Avg Return: {cat_data['avg_return']:+.1f}%
                        <br>
                        ⚠️ Risk Score: {cat_data['avg_risk_score']:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #4a1c1c; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #ff5555;">
                        <strong style="color: #ff5555;">{category}</strong> ({int(cat_data['count'])} coins)
                        <br>
                        📈 Avg Return: {cat_data['avg_return']:+.1f}%
                        <br>
                        ⚠️ Risk Score: {cat_data['avg_risk_score']:.1%}
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.info("Category data not available")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =====================
# HIGH RISK & HIGH POTENTIAL COINS
# =====================
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.markdown("### 🔥 High Risk vs 💎 High Potential Coins")

if len(df_processed) > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Top 5 High Risk Coins")
        
        high_risk_coins = df_processed[
            (df_processed['price_change_percentage_24h'] < 0) &
            (df_processed['volatility_24h'] > df_processed['volatility_24h'].median())
        ].nlargest(5, 'risk_score')
        
        if len(high_risk_coins) > 0:
            for _, coin in high_risk_coins.iterrows():
                st.markdown(f"""
                <div style="background: #4a1c1c; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #ff5555;">
                    <strong style="color: #ff5555;">{coin['symbol']}</strong> - {coin['name']}
                    <br>
                    ⚠️ Risk: <span style="color: #ff5555; font-weight: bold;">{coin['risk_score']:.1%}</span> | 
                    📉 Change: <span style="color: #ff5555;">{coin['price_change_percentage_24h']:+.1f}%</span> |
                    🔥 Vol: {(coin['volatility_24h']*100):.1f}%
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No high risk coins identified")
    
    with col2:
        st.markdown("#### 💎 Top 5 High Potential Coins")
        
        high_potential = df_processed[
            (df_processed['price_change_percentage_24h'] > 5) &
            (df_processed['volatility_24h'] < df_processed['volatility_24h'].median())
        ].nlargest(5, 'performance_score')
        
        if len(high_potential) > 0:
            for _, coin in high_potential.iterrows():
                st.markdown(f"""
                <div style="background: #1c4a2d; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #00ff88;">
                    <strong style="color: #00ff88;">{coin['symbol']}</strong> - {coin['name']}
                    <br>
                    ⭐ Potential: <span style="color: #00ff88; font-weight: bold;">{coin['performance_score']:.2f}</span> | 
                    📈 Change: <span style="color: #00ff88;">{coin['price_change_percentage_24h']:+.1f}%</span> |
                    ⚡ Vol: {(coin['volatility_24h']*100):.1f}%
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No high potential coins identified")
    
    # Risk explanation
    with st.expander("📖 Understanding Risk Scores", expanded=False):
        st.markdown("""
        <div style="background: #1a202c; padding: 15px; border-radius: 10px; border: 1px solid #2d3748;">
        **How Risk is Calculated:**
        
        1. **Volatility Component (40%)**: 
           - Based on 24-hour price range
           - Higher volatility = Higher risk score
        
        2. **Performance Component (30%)**:
           - Negative 24h returns increase risk
           - Recent underperformance is penalized
        
        3. **Market Cap Component (30%)**:
           - Smaller market caps = Higher risk
           - Larger caps generally more stable
        
        **Risk Levels:**
        - 🟢 **LOW RISK** (<30%): Stable, large-cap coins with positive returns
        - 🟡 **MEDIUM RISK** (30-70%): Moderate volatility, mixed performance
        - 🔴 **HIGH RISK** (>70%): High volatility, negative returns, small caps
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("No data available for risk analysis")

st.markdown('</div>', unsafe_allow_html=True)

# =====================
# KEY INSIGHTS & RECOMMENDATIONS
# =====================
st.markdown("---")
st.markdown('<h2 class="section-header">🎯 Key Takeaways & Recommendations</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📈 **Market Outlook**")
        
        bull_percentage = (df_processed['price_change_percentage_24h'] > 0).mean() * 100
        avg_return = df_processed['price_change_percentage_24h'].mean()
        
        st.markdown(f"""
        - ✅ **Bullish sentiment**: {bull_percentage:.0f}% coins positive
        - 📊 **Average return**: {avg_return:+.1f}%
        - 💰 **Large cap performance**: {df_processed[df_processed['market_cap_category'] == 'Large Cap']['price_change_percentage_24h'].mean():+.1f}%
        - 🎯 **Recommendation**: Consider gradual accumulation in large caps
        """)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("### ⚠️ **Risk Factors**")
        
        high_vol_coins = (df_processed['volatility_24h'] > df_processed['volatility_24h'].quantile(0.75)).sum()
        vol_percentage = (high_vol_coins / len(df_processed)) * 100
        
        st.markdown(f"""
        - 🔥 **High volatility**: {vol_percentage:.0f}% of coins (>75th percentile)
        - 📉 **Small cap volatility**: {df_processed[df_processed['market_cap_category'] == 'Small Cap']['volatility_24h'].mean()*100:.1f}% average
        - 👑 **Concentration**: Top 10 control {top_10_dominance:.1f}%
        - 🛡️ **Recommendation**: Diversify across market caps
        """)
        st.markdown('</div>', unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 💡 **Opportunities**")
        
        strong_gainers = (df_processed['price_change_percentage_24h'] > 10).sum()
        mid_cap_gainers = df_processed[
            (df_processed['market_cap_category'] == 'Mid Cap') & 
            (df_processed['price_change_percentage_24h'] > 5)
        ].shape[0]
        
        st.markdown(f"""
        - 🚀 **Strong gainers**: {strong_gainers} coins with >10% gains
        - 📈 **Mid-cap momentum**: {mid_cap_gainers} mid-caps with >5% gains
        - 💎 **High potential**: {len(high_potential) if 'high_potential' in locals() else 0} coins identified
        - 🎯 **Recommendation**: Focus on high-volume, mid-cap coins
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# =====================
# DATA TABLE FOR REFERENCE
# =====================
with st.expander("📋 **View Detailed Data Table**", expanded=False):
    st.markdown("### Detailed Cryptocurrency Data")
    
    display_cols = ['name', 'symbol', 'market_cap_formatted', 
                   'current_price', 'price_change_percentage_24h',
                   'market_cap_category', 'risk_score']
    
    available_cols = [col for col in display_cols if col in df_processed.columns]
    
    if available_cols:
        display_df = df_processed[available_cols].head(20).copy()
        
        # Format columns
        if 'current_price' in display_df.columns:
            display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
        
        if 'risk_score' in display_df.columns:
            display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.1%}")
        
        # Display with dark mode styling
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config={
                'market_cap_formatted': "Market Cap",
                'price_change_percentage_24h': st.column_config.NumberColumn(
                    "24h Change %",
                    format="%.1f%%"
                ),
                'risk_score': st.column_config.ProgressColumn(
                    "Risk Score",
                    format="%.1f%%",
                    min_value=0,
                    max_value=1
                )
            }
        )

# =====================
# PRESENTATION CONTROLS
# =====================
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎯 **Enter Presentation Mode**", use_container_width=True):
        st.success("Presentation mode activated! Press 'F' for fullscreen.")

with col2:
    if st.button("🔄 **Refresh Data**", use_container_width=True):
        st.rerun()

with col3:
    if st.button("📥 **Export Report**", use_container_width=True):
        st.info("Report export feature coming soon!")

# =====================
# FOOTER
# =====================
st.markdown(f"""
<div style="text-align: center; color: #a0aec0; padding: 20px; margin-top: 30px;">
    <hr style="border: 1px solid #2d3748; margin: 20px 0;">
    <p style="font-size: 0.9rem;">
        📊 <strong style="color: #00d4ff;">Crypto Market Intelligence Dashboard</strong> | 
        📅 Last Updated: Today | 
        🔍 {len(df_processed):,} cryptocurrencies analyzed
    </p>
    <p style="font-size: 0.8rem; color: #718096;">
        For presentation and educational purposes | Data updates in real-time
    </p>
</div>
""", unsafe_allow_html=True)
