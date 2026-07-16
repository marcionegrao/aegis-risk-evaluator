import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.decomposition import PCA

# Set page configuration
st.set_page_config(
    page_title="Aegis 🛡️ | Aligned Multi-Modal Corporate Risk Space",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern financial SaaS aesthetic
st.markdown("""
<style>
    /* Premium Pure Black Theme Override */
    .stApp {
        background: #000000 !important;
        color: #F3F4F6 !important;
    }
    
    /* Elegant HR divider */
    hr {
        border: 0 !important;
        height: 1px !important;
        background: linear-gradient(to right, rgba(139, 92, 246, 0), rgba(139, 92, 246, 0.5), rgba(139, 92, 246, 0)) !important;
        margin: 35px 0 !important;
    }
    
    /* Force high contrast white for headings & reduce sizes */
    h1, div[data-testid="stMarkdownContainer"] h1 {
        font-size: 2.2rem !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1.0rem !important;
    }
    h2, .stSubheader, [data-testid="stHeader"] h2, div[data-testid="stMarkdownContainer"] h2 {
        font-size: 1.5rem !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        margin-top: 1.8rem !important;
        margin-bottom: 1.0rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding-bottom: 8px !important;
    }
    h3, div[data-testid="stMarkdownContainer"] h3 {
        font-size: 1.25rem !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.6rem !important;
    }
    h4, h5, h6, 
    .stHeader, [data-testid="stHeader"] h3,
    div[data-testid="stMarkdownContainer"] h4,
    div[data-testid="stMarkdownContainer"] h5,
    div[data-testid="stMarkdownContainer"] h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* Force high contrast silver/off-white for all native Streamlit body text */
    div[data-testid="stMarkdownContainer"] p, 
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] ul,
    div[data-testid="stMarkdownContainer"] ol,
    div.stWrite, 
    div[data-testid="stText"] {
        color: #F3F4F6 !important; /* Extremely sharp contrast */
        font-size: 1.0rem !important;
        line-height: 1.6 !important;
    }
    
    /* Streamlit Selectbox widget labels */
    label, [data-testid="stWidgetLabel"] p, .stSelectbox label {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.0rem !important;
        letter-spacing: 0.02em !important;
        margin-bottom: 8px !important;
    }
    
    /* Complete override for Streamlit selectbox options to ensure white text */
    div[data-baseweb="select"] * {
        color: #FFFFFF !important;
    }
    
    /* Virtualized list items and option elements inside popovers */
    [data-baseweb="popover"] li, 
    [role="option"], 
    [data-baseweb="menu"] li,
    [data-baseweb="popover"] * {
        color: #FFFFFF !important;
        background-color: #11131A !important;
    }
    
    /* Ensure hover/focus states are highly visible with premium purple background and white text */
    [data-baseweb="popover"] li:hover,
    [data-baseweb="popover"] li:hover *,
    [role="option"]:hover,
    [role="option"]:hover *,
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] li:hover * {
        background-color: #8B5CF6 !important;
        color: #FFFFFF !important;
    }

    .main-header {
        font-size: 2.6rem !important; /* Reduced from 3.5rem */
        font-weight: 900;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: 0.08em;
    }
    .sub-header {
        font-size: 1.05rem !important; /* Reduced from 1.15rem */
        color: #9CA3AF !important; /* High-contrast silver subhead */
        text-align: center;
        margin-bottom: 2.5rem;
        letter-spacing: 0.04em;
    }
    .metric-card {
        background-color: #0D0E12; /* Deep Obsidian card */
        border: 1px solid rgba(255, 255, 255, 0.08); /* Thin elegant border */
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    .metric-value-healthy {
        font-size: 2.2rem;
        font-weight: 800;
        color: #10B981 !important; /* Premium Emerald */
        letter-spacing: -0.02em;
    }
    .metric-value-distressed {
        font-size: 2.2rem;
        font-weight: 800;
        color: #EF4444 !important; /* Vivid Crimson */
        letter-spacing: -0.02em;
    }
    .metric-value-caution {
        font-size: 2.2rem;
        font-weight: 800;
        color: #F59E0B !important; /* Warm Amber */
        letter-spacing: -0.02em;
    }
    .insight-card {
        background-color: #0d0f14;
        border: 1px solid rgba(139, 92, 246, 0.45); /* Muted purple border for tech vibe, more visible */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.08); /* High contrast purple glow */
    }
    .case-guide-card {
        background-color: #0D0E12 !important;
        border: 1px solid rgba(139, 92, 246, 0.25) !important;
        border-left: 5px solid #8B5CF6 !important;
        padding: 22px !important;
        border-radius: 8px !important;
        margin-bottom: 25px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.6) !important;
    }
    .case-guide-card h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        margin-top: 0 !important;
        margin-bottom: 12px !important;
    }
    .case-guide-card p, .case-guide-card div {
        color: #F3F4F6 !important; /* Extremely high-contrast off-white */
        font-size: 1.0rem !important;
        line-height: 1.6 !important;
    }
    .case-guide-card ul {
        margin-top: 8px !important;
        margin-bottom: 0 !important;
        padding-left: 20px !important;
    }
    .case-guide-card li {
        color: #F3F4F6 !important; /* Explicitly enforce high-contrast off-white for list items */
        margin-bottom: 8px !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }
    .case-guide-card li b {
        color: #FFFFFF !important; /* Pop the bold labels in white */
    }
</style>
""", unsafe_allow_html=True)

# --- CACHED DATA LOADING & SETUP ---
@st.cache_resource
def load_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load pre-calculated sequential coordinates (Dual-Tower Alignment)
    df_seq = pd.read_csv(os.path.join(base_dir, "seq_latent_coordinates.csv"))
    df_seq['CIK_int'] = df_seq['CIK'].astype(int)
    
    # Fit PCA on sequential financial coordinates
    zf_cols = [f'ZF{i}' for i in range(1, 17)]
    zt_cols = [f'ZT{i}' for i in range(1, 17)]
    pca_seq = PCA(n_components=2)
    F_2d = pca_seq.fit_transform(df_seq[zf_cols].values)
    df_seq['PCA_F1'] = F_2d[:, 0]
    df_seq['PCA_F2'] = F_2d[:, 1]
    
    T_2d = pca_seq.transform(df_seq[zt_cols].values)
    df_seq['PCA_T1'] = T_2d[:, 0]
    df_seq['PCA_T2'] = T_2d[:, 1]
    
    distressed_ciks = [886158, 1592866, 84129, 1813756, 1483510, 77149, 895126, 86874, 1005414, 13619, 13063, 20171]
    df_seq['Status'] = df_seq['CIK_int'].apply(lambda x: 'Bankrupt' if x in distressed_ciks else 'Healthy')
    
    # Load Aegis Three Scores
    df_aegis = pd.read_csv(os.path.join(base_dir, "aegis_three_scores.csv"))
    df_aegis['CIK_int'] = df_aegis['CIK'].astype(int)
    
    # Load XAI correlations
    try:
        xai_data = joblib.load(os.path.join(base_dir, "xai_correlations.pkl"))
    except Exception:
        xai_data = None
        
    return df_seq, df_aegis, xai_data, distressed_ciks, zf_cols, zt_cols, pca_seq

try:
    df_seq, df_aegis, xai_data, distressed_ciks, zf_cols, zt_cols, pca_seq = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {str(e)}")
    st.stop()

# Title banner
st.markdown("<div class='main-header'>AEGIS 🛡️</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>A Multi-Modal Architecture for Corporate Financial Distress Prediction</div>", unsafe_allow_html=True)

# Interactive Intro Description
st.markdown("""
<div class='insight-card'>
    <h3 style='color: #8B5CF6; margin-top: 0;'>🔗 Interactive Demo</h3>
    <p style='color: #E5E7EB; font-size: 1.05rem; line-height: 1.6;'>
        Welcome! This interactive dashboard demonstrates the <b>Aegis</b> dual-tower sequence alignment architecture for corporate financial distress prediction. 
        By mapping rolling 3-year trajectories of quantitative financial ratios (Financial Tower) and qualitative management disclosures (Textual Tower) into a shared 16-dimensional joint latent space, the model identifies structural risk patterns and reporting anomalies without relying on traditional supervised bankruptcy labels.
    </p>
    <p style='color: #E5E7EB; font-size: 0.95rem; margin-bottom: 0;'>
        💡 <b>How to interact:</b> Select a historical company profile below to visualize how its financial trajectory (yellow path) and qualitative management disclosures (blue path) evolved and aligned over consecutive reporting periods.
    </p>
</div>
""", unsafe_allow_html=True)

# Curated LinkedIn Demo Case Studies Selection
demo_cases = {
    "PARTY CITY HOLDCO INC. (Multi-Modal Decoupling Case)": "PARTY CITY HOLDCO INC.",
    "RITE AID CORP (Symmetric Distress Case)": "RITE AID CORP",
    "CHESAPEAKE ENERGY CORP (High-Velocity Distress Case)": "CHESAPEAKE ENERGY CORP",
    "CATALENT, INC. (Early Warning Narrative Divergence Case)": "CATALENT, INC.",
    "NVIDIA CORP (Symmetric Healthy Alignment Case)": "NVIDIA CORP"
}

st.markdown("---")
st.subheader("🏢 Select Case Study")
selected_demo_label = st.selectbox("Choose a High-Profile Corporate Trajectory to Plot", list(demo_cases.keys()), label_visibility="collapsed")
selected_company = demo_cases[selected_demo_label]

# Map Selected Company to CIK
seq_companies = df_seq[['CIK_int', 'Company_Name']].drop_duplicates()
selected_cik = seq_companies[seq_companies['Company_Name'] == selected_company]['CIK_int'].values[0]

# Demo Explanatory Guide Content
case_guides = {
    "PARTY CITY HOLDCO INC.": {
        "title": "🚨 SEVERE DECOUPLING: PARTY CITY HOLDCO INC.",
        "icon": "🚨",
        "desc": """
        <b>The Scenario:</b> Party City, carrying high leverage, faced sudden operational disruption during the 2020 retail lockdowns.
        <br><br>
        <b>The Trajectory Analysis:</b>
        <ul>
            <li><b>2019 (Aligned Stability):</b> Financial trajectories and qualitative narrative disclosures were tightly aligned. The Latent Gap was minimal (0.211), indicating that executive commentary closely matched underlying financial metrics.</li>
            <li><b>2020 (The Decoupling):</b> When operational revenues fell, the financial coordinates shifted deeply into the distressed region (Score 1 Financial: <b>-0.9657</b>). However, qualitative text disclosures remained defensive and highly positive (Score 1 Text: <b>+0.9656</b>).</li>
            <li><b>Aegis Signal:</b> The Latent Gap expanded to <b>1.5585</b>, triggering a <b>🚨 SEVERE DECOUPLING (Score 3)</b> alert. This indicates a significant reporting-financial disconnect or reporting lag up to 12 months prior to formal restructuring.</li>
        </ul>
        """
    },
    "RITE AID CORP": {
        "title": "🔴 CHRONIC DISTRESS & SYMMETRIC DECLINE: RITE AID CORP",
        "icon": "🔴",
        "desc": """
        <b>The Scenario:</b> The legacy drugstore chain faced high debt levels, severe competitive pressure, and significant legal liabilities.
        <br><br>
        <b>The Trajectory Analysis:</b>
        <ul>
            <li><b>The Path:</b> Unlike sudden macroeconomic shocks, Rite Aid illustrates a multi-year chronic decline. Both financial and qualitative trajectories drift in lockstep toward the distressed region of the latent space.</li>
            <li><b>Aegis Signal:</b> Financial and Textual Position Scores are in <b>Symmetric Alignment</b> but located deep in negative territory. This indicates no significant reporting anomaly; executive commentary transparently reflected the ongoing financial challenges.</li>
        </ul>
        """
    },
    "CHESAPEAKE ENERGY CORP": {
        "title": "⚡ HIGH-VELOCITY COLLAPSE: CHESAPEAKE ENERGY CORP",
        "icon": "⚡",
        "desc": """
        <b>The Scenario:</b> The natural gas producer was impacted by falling energy commodity prices combined with a debt-burdened balance sheet.
        <br><br>
        <b>The Trajectory Analysis:</b>
        <ul>
            <li><b>The Path:</b> Note the high <b>Velocity (Score 2)</b> in both towers as their coordinates shift rapidly across the latent space from 2018 to 2020.</li>
            <li><b>Aegis Signal:</b> The rapid velocity and downward trajectory indicate an acute liquidity and solvency crisis, culminating in a Chapter 11 filing.</li>
        </ul>
        """
    },
    "CATALENT, INC.": {
        "title": "⚠️ EARLY WARNING DIVERGENCE: CATALENT, INC.",
        "icon": "⚠️",
        "desc": """
        <b>The Scenario:</b> The contract pharmaceutical manufacturer faced operational bottlenecks and quality-control constraints at key sterile manufacturing facilities.
        <br><br>
        <b>The Trajectory Analysis:</b>
        <ul>
            <li><b>The Path:</b> In 2022, while traditional trailing financial ratios appeared stable and healthy, qualitative narrative disclosures began highlighting operational risks and production bottlenecks.</li>
            <li><b>Aegis Signal:</b> Triggered a <b>Critical Negative Divergence (Smoke Signal)</b>. The narrative score deteriorated sharply ahead of trailing financial metrics, demonstrating the value of qualitative disclosures as leading indicators.</li>
        </ul>
        """
    },
    "NVIDIA CORP": {
        "title": "🟢 SYMMETRIC ALIGNMENT (HEALTHY): NVIDIA CORP",
        "icon": "🟢",
        "desc": """
        <b>The Scenario:</b> The market leader in GPU computing and artificial intelligence infrastructure.
        <br><br>
        <b>The Trajectory Analysis:</b>
        <ul>
            <li><b>The Path:</b> Both financial and narrative trajectories remain tightly clustered and nested deep within the <b>"Continent of Normalcy"</b> (the safe, stable region).</li>
            <li><b>Aegis Signal:</b> Shows near-perfect symmetric alignment, stable near-zero velocity, and negligible decoupling distance, reflecting excellent operational performance and high reporting transparency.</li>
        </ul>
        """
    }
}

# Render Case Guide
guide = case_guides[selected_company]
st.markdown(f"""
<div class='case-guide-card'>
    <h3>{guide['icon']} {guide['title']}</h3>
    <div>{guide['desc']}</div>
</div>
""", unsafe_allow_html=True)

# --- PLOT DUAL-TRAJECTORY SCATTER ---
# Subsample background points for quick rendering
healthy_seq = df_seq[df_seq['Status'] == 'Healthy']
bankrupt_seq = df_seq[df_seq['Status'] == 'Bankrupt']

healthy_seq_sample = healthy_seq.sample(n=min(1200, len(healthy_seq)), random_state=42)
plot_seq_df = pd.concat([healthy_seq_sample, bankrupt_seq], ignore_index=True)

# Isolate target company sequential history
target_seq = df_seq[df_seq['CIK_int'] == selected_cik].copy().sort_values('Filing_Year_End')
plot_seq_df['Display_Status'] = plot_seq_df['Status']

# Generate the dual trajectory scatter plot
fig_seq = px.scatter(
    plot_seq_df, x='PCA_F1', y='PCA_F2',
    color='Display_Status',
    hover_data=['Company_Name', 'Filing_Year_End'],
    color_discrete_map={
        'Healthy': 'rgba(31, 119, 180, 0.1)', # highly transparent blue
        'Bankrupt': 'rgba(214, 39, 40, 0.6)' # transparent red
    },
    title=f"Aegis Latent Space: Aligned Dual-Trajectory of {selected_company}",
    labels={'PCA_F1': 'Aligned Dimension 1 (PCA1)', 'PCA_F2': 'Aligned Dimension 2 (PCA2)', 'Display_Status': 'Firm Status'},
    render_mode='webgl'
)

fig_seq.update_layout(
    template="plotly_dark",
    height=600,
    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title_font=dict(color='#FFFFFF'), tickfont=dict(color='#E5E7EB')),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title_font=dict(color='#FFFFFF'), tickfont=dict(color='#E5E7EB')),
    title_font=dict(color='#FFFFFF'),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(
        font=dict(color="#FFFFFF", size=12),
        bgcolor="rgba(10,10,15,0.8)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
        title=dict(font=dict(color="#FFFFFF", size=13))
    )
)

# Plot paths if we have records
if len(target_seq) > 0:
    # 1. Add Financial Path (Yellow/Gold)
    fig_seq.add_trace(go.Scatter(
        x=target_seq['PCA_F1'],
        y=target_seq['PCA_F2'],
        mode='lines+markers+text',
        name='Financial Trajectory (ZF)',
        text=target_seq['Filing_Year_End'].astype(str),
        textposition="top center",
        line=dict(color='#ffb86c', width=4),
        marker=dict(color='#ff79c6', size=11, symbol='circle')
    ))
    
    # 2. Add Textual Path (Cyan)
    fig_seq.add_trace(go.Scatter(
        x=target_seq['PCA_T1'],
        y=target_seq['PCA_T2'],
        mode='lines+markers+text',
        name='Narrative Trajectory (ZT)',
        text=target_seq['Filing_Year_End'].astype(str) + " Text",
        textposition="bottom center",
        line=dict(color='#8be9fd', width=3, dash='dash'),
        marker=dict(color='#bd93f9', size=11, symbol='diamond')
    ))

# Better explain the graph with clear interactive directions
st.markdown("""
<div style='background-color: #11131A; border: 1px solid rgba(139, 92, 246, 0.2); padding: 18px; border-radius: 8px; margin-bottom: 20px;'>
    <h4 style='color: #8be9fd !important; margin-top: 0; margin-bottom: 10px;'>💡 Understanding & Interacting with the Latent Space Map</h4>
    <p style='color: #F3F4F6 !important; font-size: 1.0rem; line-height: 1.6; margin-bottom: 0;'>
        This chart projects Aegis's high-dimensional 16D space into 2D using Principal Component Analysis (PCA). 
        The small background dots are benchmark companies, mapping out the <b>Continent of Normalcy</b> (healthy stable firms in transparent blue) and the <b>Cliff Edge</b> (distressed firms in transparent red).
        <br><br>
        <b>How to Interact:</b>
        <br>
        • 🔍 <b>Zooming & Panning:</b> Click and drag a rectangle over any area of the plot to zoom in and examine specific coordinates. Double-click anywhere on the plot to reset the zoom view. You can also zoom with your mouse wheel or trackpad.
        <br>
        • 📌 <b>Isolate & Highlight Lines:</b> To highlight or hide specific lines (e.g., to focus only on the narrative or financial line), single-click the legend labels on the right (<i>Financial Trajectory (ZF)</i> or <i>Narrative Trajectory (ZT)</i>).
        <br>
        • ℹ️ <b>Hover Details:</b> Hover over any of the gold circles or blue diamonds to see the exact filing year, company name, and latent space coordinates.
        <br>
        • 🔗 <b>Identifying Decoupling Gaps:</b> The solid gold line shows the financial trajectory, and the dashed cyan line shows the narrative trajectory. When these lines separate and head in different directions, it signals cross-modal reporting anomalies and potential distress.
    </p>
</div>
""", unsafe_allow_html=True)
    
st.plotly_chart(fig_seq, use_container_width=True)

st.markdown("---")
# --- DISPLAY METRICS AND THREE-SCORE ANALYSIS ---
st.subheader(f"📊 Aegis Three-Score Diagnostics: {selected_company}")

# Filter the Aegis Scores for our selected company
company_aegis = df_aegis[df_aegis['CIK_int'] == selected_cik].sort_values('Filing_Year_End')

if len(company_aegis) > 0:
    table_df = company_aegis[[
        'Filing_Year_End', 'Score_1_Fin', 'Score_1_Text', 
        'Score_2_Fin_Velocity', 'Score_2_Fin_Directional',
        'Score_3_Decoupling_Distance', 'Score_3_Signed_Divergence'
    ]].copy()
    
    table_df.columns = [
        'Filing Year', 'Score 1: Fin Position', 'Score 1: Text Position',
        'Score 2: Fin Velocity', 'Score 2: Fin Direction',
        'Score 3: Latent Gap', 'Score 3: Signed Divergence'
    ]
    
    st.dataframe(table_df.set_index('Filing Year').style.format({
        'Score 1: Fin Position': '{:+.4f}',
        'Score 1: Text Position': '{:+.4f}',
        'Score 2: Fin Velocity': '{:.4f}',
        'Score 2: Fin Direction': '{:+.4f}',
        'Score 3: Latent Gap': '{:.4f}',
        'Score 3: Signed Divergence': '{:+.4f}'
    }), use_container_width=True)
    
    # --- Trajectory & Divergence Dashboards ---
    st.markdown("### 📈 Time-Series Diagnostic Dashboards")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("<h5 style='text-align: center; color: #bd93f9; margin-bottom: 10px;'>Health Position Spectrum Over Time</h5>", unsafe_allow_html=True)
        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(
            x=company_aegis['Filing_Year_End'],
            y=company_aegis['Score_1_Fin'],
            mode='lines+markers',
            name='Financial (ZF)',
            line=dict(color='#ffb86c', width=3.5),
            marker=dict(size=9, symbol='circle', color='#ff79c6')
        ))
        fig_traj.add_trace(go.Scatter(
            x=company_aegis['Filing_Year_End'],
            y=company_aegis['Score_1_Text'],
            mode='lines+markers',
            name='Narrative (ZT)',
            line=dict(color='#8be9fd', width=3.5, dash='dash'),
            marker=dict(size=9, symbol='diamond', color='#bd93f9')
        ))
        fig_traj.update_layout(
            template="plotly_dark",
            height=320,
            xaxis=dict(
                title="Filing Year",
                tickmode='array',
                tickvals=company_aegis['Filing_Year_End'].tolist(),
                showgrid=True,
                gridcolor='rgba(255,255,255,0.05)',
                title_font=dict(color='#FFFFFF'),
                tickfont=dict(color='#E5E7EB')
            ),
            yaxis=dict(
                title="Health Score (-1 to +1)",
                range=[-1.05, 1.05],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.05)',
                title_font=dict(color='#FFFFFF'),
                tickfont=dict(color='#E5E7EB')
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#FFFFFF", size=11)
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, b=30, l=30, r=30)
        )
        st.plotly_chart(fig_traj, use_container_width=True)
        
    with col_chart2:
        st.markdown("<h5 style='text-align: center; color: #8be9fd; margin-bottom: 10px;'>Signed Divergence Gap (Score 3)</h5>", unsafe_allow_html=True)
        div_colors = ['#ef4444' if val < -0.15 else '#10b981' if val > 0.15 else '#8F9CAE' for val in company_aegis['Score_3_Signed_Divergence']]
        fig_div = go.Figure()
        fig_div.add_trace(go.Bar(
            x=company_aegis['Filing_Year_End'],
            y=company_aegis['Score_3_Signed_Divergence'],
            marker_color=div_colors,
            text=company_aegis['Score_3_Signed_Divergence'].round(4),
            textposition='auto',
            name='Signed Divergence'
        ))
        fig_div.update_layout(
            template="plotly_dark",
            height=320,
            xaxis=dict(
                title="Filing Year",
                tickmode='array',
                tickvals=company_aegis['Filing_Year_End'].tolist(),
                showgrid=False,
                title_font=dict(color='#FFFFFF'),
                tickfont=dict(color='#E5E7EB')
            ),
            yaxis=dict(
                title="Signed Divergence",
                showgrid=True,
                gridcolor='rgba(255,255,255,0.05)',
                title_font=dict(color='#FFFFFF'),
                tickfont=dict(color='#E5E7EB')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, b=30, l=30, r=30)
        )
        st.plotly_chart(fig_div, use_container_width=True)
    
    st.markdown("### 🔬 Period-Specific Deep Dive")
    selected_year = st.selectbox("Select Diagnostic Reporting Year", company_aegis['Filing_Year_End'].tolist())
    
    year_row = company_aegis[company_aegis['Filing_Year_End'] == selected_year].iloc[0]
    
    s1_fin = year_row['Score_1_Fin']
    s1_text = year_row['Score_1_Text']
    s2_vel = year_row['Score_2_Fin_Velocity']
    s2_dir = year_row['Score_2_Fin_Directional']
    s3_gap = year_row['Score_3_Decoupling_Distance']
    s3_div = year_row['Score_3_Signed_Divergence']
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown(f"""
        <div style='background-color:#0b0c10; border: 1px solid rgba(255,255,255,0.08); padding: 20px; border-radius: 12px; height: 260px; text-align: center;'>
            <h4 style='color:#bd93f9 !important; margin-top:0;'>SCORE 1: LATENT POSITION</h4>
            <p style='font-size: 0.85rem; color:#E5E7EB !important;'>Spectrum of Health (-1.0 to +1.0)</p>
            <div style='display: flex; justify-content: space-around; margin-top:20px;'>
                <div>
                    <span style='font-size:0.8rem; color:#8be9fd !important;'>Financial (ZF)</span>
                    <div style='font-size: 1.8rem; font-weight:800; color:{"#ef4444" if s1_fin < -0.1 else "#10b981" if s1_fin > 0.1 else "#f59e0b"} !important;'>{s1_fin:+.4f}</div>
                </div>
                <div>
                    <span style='font-size:0.8rem; color:#ff79c6 !important;'>Textual (ZT)</span>
                    <div style='font-size: 1.8rem; font-weight:800; color:{"#ef4444" if s1_text < -0.1 else "#10b981" if s1_text > 0.1 else "#f59e0b"} !important;'>{s1_text:+.4f}</div>
                </div>
            </div>
            <div style='margin-top: 15px; font-size: 0.85rem; color:#E5E7EB !important;'>
                <span>Position Status: </span>
                <b>{"🟢 HEALTHY" if s1_fin > 0.1 else "🔴 CRITICAL DISTRESS" if s1_fin < -0.1 else "🟡 TRANSITION ZONE"}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_s2:
        st.markdown(f"""
        <div style='background-color:#0b0c10; border: 1px solid rgba(255,255,255,0.08); padding: 20px; border-radius: 12px; height: 260px; text-align: center;'>
            <h4 style='color:#ffb86c !important; margin-top:0;'>SCORE 2: YoY MOMENTUM</h4>
            <p style='font-size: 0.85rem; color:#E5E7EB !important;'>Velocity & Trajectory Change</p>
            <div style='display: flex; justify-content: space-around; margin-top:20px;'>
                <div>
                    <span style='font-size:0.8rem; color:#8be9fd !important;'>Speed (16D Norm)</span>
                    <div style='font-size: 1.8rem; font-weight:800; color:#ffb86c !important;'>{s2_vel:.4f}</div>
                </div>
                <div>
                    <span style='font-size:0.8rem; color:#ff79c6 !important;'>Direction (Trend)</span>
                    <div style='font-size: 1.8rem; font-weight:800; color:{"#10b981" if s2_dir > 0.1 else "#ef4444" if s2_dir < -0.1 else "#E5E7EB"} !important;'>{s2_dir:+.4f}</div>
                </div>
            </div>
            <div style='margin-top: 15px; font-size: 0.85rem; color:#E5E7EB !important;'>
                <span>Momentum Status: </span>
                <b>{"🟢 IMPROVING" if s2_dir > 0.1 else "🔴 DECAYING" if s2_dir < -0.1 else "⚪ STABLE"}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_s3:
        st.markdown(f"""
        <div style='background-color:#0b0c10; border: 1px solid rgba(255,255,255,0.08); padding: 20px; border-radius: 12px; height: 260px; text-align: center;'>
            <h4 style='color:#8be9fd !important; margin-top:0;'>SCORE 3: DECOUPLING</h4>
            <p style='font-size: 0.85rem; color:#E5E7EB !important;'>Multi-Modal Vector Alignment</p>
            <div style='display: flex; justify-content: space-around; margin-top:20px;'>
                <div>
                    <span style='font-size:0.8rem; color:#8be9fd !important;'>Latent Gap</span>
                    <div style='font-size: 1.8rem; font-weight:800; color:#bd93f9 !important;'>{s3_gap:.4f}</div>
                </div>
                <div>
                    <span style='font-size:0.8rem; color:#ff79c6 !important;'>Signed Div.</span>
                    <div style='font-size: 1.8rem; font-weight:800; color:{"#ef4444" if abs(s3_div) > 0.3 else "#f59e0b" if abs(s3_div) > 0.15 else "#10b981"} !important;'>{s3_div:+.4f}</div>
                </div>
            </div>
            <div style='margin-top: 15px; font-size: 0.85rem; color:#E5E7EB !important;'>
                <span>Alignment: </span>
                <b>{"🚨 CRITICAL MISALIGNMENT" if abs(s3_div) > 0.3 else "⚠️ MODERATE MISALIGNMENT" if abs(s3_div) > 0.15 else "✅ SYMMETRIC"}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Multi-dimensional interpretation of decoupling
    st.markdown("<br>", unsafe_allow_html=True)
    if abs(s3_div) <= 0.15:
        st.success(f"### ✅ **Symmetric Multi-Modal Integrity Intact**\n**Interpretation:** Aligned (Narrative and financials are in harmony). Management's reports and alternative operational metrics perfectly match the hard financial numbers (Signed Divergence: {s3_div:+.4f}).")
    elif s3_div > 0.15:
        if s1_fin > 0 and s1_text < 0:
            st.warning(f"### ⚠️ **CRITICAL NEGATIVE DIVERGENCE (Smoke Signal)**\n**Interpretation:** Financials look healthy ({s1_fin:+.4f}), but operational/text metrics are in distress ({s1_text:+.4f})! This indicates standard trailing financial metrics look stable, but real-time narrative or alternative indicators are deteriorating rapidly. This is a primary early warning indicator!")
        else:
            st.warning(f"### ⚠️ **NEGATIVE DIVERGENCE**\n**Interpretation:** Financial metrics are outpacing operational/narrative metrics. The balance sheet numbers have expanded faster than narrative-based context has caught up (Signed Divergence: {s3_div:+.4f}).")
    else: # s3_div < -0.15
        if s1_fin < 0 and s1_text > 0:
            st.error(f"### 🚨 **SEVERE DECOUPLING / NARRATIVE DISCONNECT**\n**Interpretation:** Financial statements are in deep distress ({s1_fin:+.4f}), but operational narrative remains highly optimistic or lagging ({s1_text:+.4f})! This indicates the corporate books are in the gutter, but management's report or alternative data hasn't acknowledged it yet (Signed Divergence: {s3_div:+.4f}). This is a classic indicator of executive over-optimism or reporting lag.")
        else:
            st.info(f"### 🟢 **POSITIVE DIVERGENCE (Turnaround Signal)**\n**Interpretation:** Operational/narrative metrics are improving ahead of trailing financial statements. Qualitative improvements or leading alternative indicators (hiring, bill payment, traffic) are turning around before the trailing 12-month financials reflect recovery (Signed Divergence: {s3_div:+.4f}). This is an early-stage turnaround indicator!")

# --- 🛡️ AEGIS EXPLAINABLE AI (XAI) LATENT INTERPRETER ---
if xai_data is not None and len(target_seq) > 0:
    st.markdown("---")
    st.subheader("🔍 Aegis Explainable AI (XAI) Latent Interpreter")
    st.write("""
    Aegis maps financial ratios and qualitative narratives into 16 joint latent dimensions. 
    Select a year below to dissect **exactly** which financial ratios and narrative concepts are driving the company's high-dimensional coordinates during that period.
    """)
    
    selected_xai_year = st.selectbox("Select Year to Explain", target_seq['Filing_Year_End'].tolist(), key="xai_year")
    
    year_rows = target_seq[target_seq['Filing_Year_End'] == selected_xai_year]
    if len(year_rows) > 0:
        row_target = year_rows.iloc[0]
        
        # Show coordinates expander
        with st.expander("🌐 Raw 16D Aligned Coordinates (ZF & ZT)"):
            st.write("Each coordinate represents the position on the unit sphere along that specific axis (range -1.0 to 1.0).")
            coord_df = pd.DataFrame({
                'Dimension': [f'Dim {i}' for i in range(1, 17)],
                'Financial Coordinate (ZF)': [row_target[f'ZF{i}'] for i in range(1, 17)],
                'Text Coordinate (ZT)': [row_target[f'ZT{i}'] for i in range(1, 17)]
            })
            st.dataframe(coord_df.set_index('Dimension'), use_container_width=True)
            
        # Find top 3 most active dimensions (highest absolute value of financial coordinate)
        zf_values = [row_target[f'ZF{i}'] for i in range(1, 17)]
        abs_values = [abs(v) for v in zf_values]
        top_active_idx = np.argsort(abs_values)[::-1][:3]
        
        st.markdown("### 🧬 Top 3 Latent Drivers for this Period")
        st.write("These represent the three most heavily activated coordinates in the company's 16-dimensional risk state:")
        
        cols_active = st.columns(3)
        for idx_col, dim_idx in enumerate(top_active_idx):
            dim_num = dim_idx + 1
            val = zf_values[dim_idx]
            
            with cols_active[idx_col]:
                card_color = "#ef4444" if val < -0.2 else "#10b981" if val > 0.2 else "#f59e0b"
                st.markdown(f"""
                <div style='background-color:#11131A; border: 1px solid rgba(255,255,255,0.08); padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 10px;'>
                    <h4 style='color:#bd93f9 !important; margin-bottom: 5px; margin-top: 0;'>Dimension {dim_num}</h4>
                    <p style='font-size: 1.8rem; font-weight: 700; color:{card_color} !important; margin: 0;'>{val:.4f}</p>
                    <p style='font-size: 0.85rem; color:#8be9fd !important; margin: 0;'>Active Driver</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Get top financial correlations for this dimension
                fin_corr_col = xai_data['fin_corr'][f'ZF{dim_num}']
                if val >= 0:
                    top_ratios = fin_corr_col.sort_values(ascending=False).head(3)
                    st.markdown("**Top Supporting Ratios (Positive):**")
                else:
                    top_ratios = fin_corr_col.sort_values(ascending=True).head(3)
                    st.markdown("**Top Supporting Ratios (Negative):**")
                    
                for r_name, r_val in top_ratios.items():
                    st.write(f"- `{r_name}` ({r_val:+.2f})")
                    
                # Get top textual correlations for this dimension
                text_corr_col = xai_data['text_corr'][f'ZT{dim_num}']
                if val >= 0:
                    top_themes = text_corr_col.sort_values(ascending=False).head(2)
                else:
                    top_themes = text_corr_col.sort_values(ascending=True).head(2)
                    
                st.markdown("<br>**Associated Narrative Themes:**", unsafe_allow_html=True)
                for t_name, t_val in top_themes.items():
                    top_words = xai_data['theme_words'][t_name][:5]
                    st.markdown(f"- **{t_name.replace('_', ' ')}** ({t_val:+.2f})")
                    st.caption(f"  *Key words:* {', '.join(top_words)}")


