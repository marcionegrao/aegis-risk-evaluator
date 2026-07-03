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
    }
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 0.1rem;
        letter-spacing: 0.08em;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #9CA3AF; /* Muted gray subhead */
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
        font-size: 2.5rem;
        font-weight: 800;
        color: #10B981; /* Premium Emerald */
        letter-spacing: -0.02em;
    }
    .metric-value-distressed {
        font-size: 2.5rem;
        font-weight: 800;
        color: #EF4444; /* Vivid Crimson */
        letter-spacing: -0.02em;
    }
    .metric-value-caution {
        font-size: 2.5rem;
        font-weight: 800;
        color: #F59E0B; /* Warm Amber */
        letter-spacing: -0.02em;
    }
    .insight-card {
        background-color: #11131A;
        border: 1px solid rgba(139, 92, 246, 0.25); /* Muted purple border for tech vibe */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
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
st.markdown("<div class='sub-header'>A Multi-Modal, Deep Trajectory Alignment Architecture for Corporate Financial Distress Prediction</div>", unsafe_allow_html=True)

# Interactive Intro Description
st.markdown("""
<div class='insight-card'>
    <h3 style='color: #8B5CF6; margin-top: 0;'>🔗 Curated Interactive LinkedIn Demo</h3>
    <p style='color: #E5E7EB; font-size: 1.05rem; line-height: 1.6;'>
        Welcome! This demo showcases <b>Aegis</b>, an enterprise-grade AI risk-evaluator designed to catch corporate credit distress and financial reporting anomalies. 
        Instead of relying on outdated 1960s linear formulas (like the Altman Z-score), Aegis utilizes an unsupervised, deep PyTorch parallel <b>Dual-Tower LSTM architecture</b>.
        It maps 3-year sequential financial ratios (Financial Tower) and corresponding 10-K qualitative risk narratives (Textual Tower) into the <b>exact same 16-dimensional joint latent space</b>.
    </p>
    <p style='color: #9CA3AF; font-size: 0.95rem; margin-bottom: 0;'>
        💡 <b>How to interact:</b> Choose a curated company profile below to watch how its actual financials (yellow path) and management narrative disclosures (blue path) move relative to each other over consecutive years in high-dimensional space.
    </p>
</div>
""", unsafe_allow_html=True)

# Curated LinkedIn Demo Case Studies Selection
demo_cases = {
    "PARTY CITY HOLDCO INC. (Classic Pandemic Decoupling)": "PARTY CITY HOLDCO INC.",
    "RITE AID CORP (Chronic Distress & Opioid Liability Decay)": "RITE AID CORP",
    "CHESAPEAKE ENERGY CORP (Pure Balance-Sheet Collapse)": "CHESAPEAKE ENERGY CORP",
    "CATALENT, INC. (Wegovy Syringe Factory Crisis & Negative Divergence)": "CATALENT, INC.",
    "NVIDIA CORP (Symmetric Alignment in Continent of Normalcy)": "NVIDIA CORP"
}

st.subheader("🏢 Select Curated Case Study")
selected_demo_label = st.selectbox("Choose a High-Profile Corporate Trajectory to Plot", list(demo_cases.keys()))
selected_company = demo_cases[selected_demo_label]

# Map Selected Company to CIK
seq_companies = df_seq[['CIK_int', 'Company_Name']].drop_duplicates()
selected_cik = seq_companies[seq_companies['Company_Name'] == selected_company]['CIK_int'].values[0]

# Demo Explanatory Guide Content
case_guides = {
    "PARTY CITY HOLDCO INC.": {
        "title": "🚨 SEVERE DECOUPLING ALERT: PARTY CITY HOLDCO INC.",
        "icon": "🚨",
        "desc": """
        <b>The Scenario:</b> Party City carrying high debt faced catastrophic brick-and-mortar store closures during the 2020 COVID-19 lockdowns.
        <br><br>
        <b>The Trajectory Analysis:</b>
        <ul>
            <li><b>2019 (Aligned Stability):</b> Financials and qualitative narratives are tightly aligned. The Latent Gap is extremely small (0.211), showing that management's narrative is in full consensus with actual financial fundamentals.</li>
            <li><b>2020 (The Decoupling):</b> When seasonal sales crashed, the company's financial coordinates plummeted into deep distress (Score 1 Financial: <b>-0.9657</b>). However, executive text disclosures remained highly positive and defensive (Score 1 Text: <b>+0.9656</b>).</li>
            <li><b>Aegis Signal:</b> The Latent Gap exploded to <b>1.5585</b>, triggering a massive <b>🚨 SEVERE DECOUPLING (Score 3)</b> alert, flagging delusional management disclosures or reporting lag 12 months before filing for bankruptcy.</li>
        </ul>
        """
    },
    "RITE AID CORP": {
        "title": "🔴 CHRONIC DISTRESS & SYMMETRIC DECLINE: RITE AID CORP",
        "icon": "🔴",
        "desc": """
        <b>The Scenario:</b> The legacy drugstore giant was weighed down by massive debt, rising retail competition, and billions in opioid lawsuit liabilities.
        <br><br>
        <b>The Trajectory Analysis:</b>
        <ul>
            <li><b>The Path:</b> Unlike sudden shocks, Rite Aid is a textbook case of a chronic decline. Watch both trajectories over consecutive years drift in lockstep toward the distressed region (Cliff Edge).</li>
            <li><b>Aegis Signal:</b> Score 1 Financial and Score 1 Text are in <b>Symmetric Alignment</b> but both deep in the negative territory. This confirms that there is no reporting anomaly; management’s disclosures are transparently reflecting the severe financial erosion of the business.</li>
        </ul>
        """
    },
    "CHESAPEAKE ENERGY CORP": {
        "title": "⚡ HIGH-VELOCITY COLLAPSE: CHESAPEAKE ENERGY CORP",
        "icon": "⚡",
        "desc": """
        <b>The Scenario:</b> The shale gas pioneer was hit by crashes in natural gas prices and a debt-fueled balance sheet that was highly unsustainable.
        <br><br>
        <b>The Trajectory Analysis:</b>
        <ul>
            <li><b>The Path:</b> Notice the extremely high <b>Velocity (Score 2)</b> in both towers as their coordinates leap across the latent space from 2018 to 2020.</li>
            <li><b>Aegis Signal:</b> The rapid rate of displacement (high velocity) coupled with deep negative direction indicates a sudden, unavoidable solvency crisis, culminating in a swift chapter 11 filing in June 2020.</li>
        </ul>
        """
    },
    "CATALENT, INC.": {
        "title": "⚠️ EARLY WARNING SMOKE SIGNAL: CATALENT, INC.",
        "icon": "⚠️",
        "desc": """
        <b>The Scenario:</b> The pharmaceutical manufacturer struggled with operational bottlenecks and sterile syringe quality control issues at key plants producing Wegovy weight-loss drug.
        <br><br>
        <b>The Trajectory Analysis:</b>
        <ul>
            <li><b>The Path:</b> In 2022, while trailing financial ratios looked stable and healthy, the qualitative narrative disclosures in filings began expressing deep risk, supply-chain delays, and quality-control bottlenecks.</li>
            <li><b>Aegis Signal:</b> Triggered a <b>Critical Negative Divergence ("Smoke Signal")</b>. The narrative score deteriorated sharply ahead of trailing balance sheet books. This proved that alternative disclosures are leading risk indicators.</li>
        </ul>
        """
    },
    "NVIDIA CORP": {
        "title": "🟢 SYMMETRIC ALIGNMENT (HEALTHY): NVIDIA CORP",
        "icon": "🟢",
        "desc": """
        <b>The Scenario:</b> The sovereign leader of GPU silicon and artificial intelligence infrastructure.
        <br><br>
        <b>The Trajectory Analysis:</b>
        <ul>
            <li><b>The Path:</b> Both the Financial and Textual trajectories remain tightly clustered and nested deep within the <b>"Continent of Normalcy"</b> (the safe region).</li>
            <li><b>Aegis Signal:</b> Shows near-perfect symmetric alignment, stable near-zero velocity, and negligible decoupling distance. The narrative and numbers reflect unparalleled operational strength and ideal reporting integrity.</li>
        </ul>
        """
    }
}

# Render Case Guide
guide = case_guides[selected_company]
st.markdown(f"""
<div style='background-color: #0F1115; border-left: 5px solid #8B5CF6; padding: 20px; border-radius: 4px; margin-bottom: 25px;'>
    <h3 style='color: #FFFFFF; margin-top: 0; margin-bottom: 8px;'>{guide['icon']} {guide['title']}</h3>
    <p style='color: #D1D5DB; font-size: 1.0rem; line-height: 1.6; margin: 0;'>{guide['desc']}</p>
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
    labels={'PCA_F1': 'Aligned Dimension 1 (PCA1)', 'PCA_F2': 'Aligned Dimension 2 (PCA2)'},
    render_mode='webgl'
)

fig_seq.update_layout(
    template="plotly_dark",
    height=600,
    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
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
    
st.plotly_chart(fig_seq, use_container_width=True)

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
                gridcolor='rgba(255,255,255,0.05)'
            ),
            yaxis=dict(
                title="Health Score (-1 to +1)",
                range=[-1.05, 1.05],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.05)'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, b=30, l=30, r=30)
        )
        st.plotly_chart(fig_traj, use_container_width=True)
        
    with col_chart2:
        st.markdown("<h5 style='text-align: center; color: #8be9fd; margin-bottom: 10px;'>Signed Divergence Gap (Score 3)</h5>", unsafe_allow_html=True)
        div_colors = ['#ef4444' if val < -0.15 else '#10b981' if val > 0.15 else '#9ca3af' for val in company_aegis['Score_3_Signed_Divergence']]
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
                showgrid=False
            ),
            yaxis=dict(
                title="Signed Divergence",
                showgrid=True,
                gridcolor='rgba(255,255,255,0.05)'
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
            <h4 style='color:#bd93f9; margin-top:0;'>SCORE 1: LATENT POSITION</h4>
            <p style='font-size: 0.85rem; color:#9ca3af;'>Spectrum of Health (-1.0 to +1.0)</p>
            <div style='display: flex; justify-content: space-around; margin-top:20px;'>
                <div>
                    <span style='font-size:0.8rem; color:#8be9fd;'>Financial (ZF)</span>
                    <div style='font-size: 1.8rem; font-weight:800; color:{"#ef4444" if s1_fin < -0.1 else "#10b981" if s1_fin > 0.1 else "#f59e0b"};'>{s1_fin:+.4f}</div>
                </div>
                <div>
                    <span style='font-size:0.8rem; color:#ff79c6;'>Textual (ZT)</span>
                    <div style='font-size: 1.8rem; font-weight:800; color:{"#ef4444" if s1_text < -0.1 else "#10b981" if s1_text > 0.1 else "#f59e0b"};'>{s1_text:+.4f}</div>
                </div>
            </div>
            <div style='margin-top: 15px; font-size: 0.85rem;'>
                <span>Position Status: </span>
                <b>{"🟢 HEALTHY" if s1_fin > 0.1 else "🔴 CRITICAL DISTRESS" if s1_fin < -0.1 else "🟡 TRANSITION ZONE"}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_s2:
        st.markdown(f"""
        <div style='background-color:#0b0c10; border: 1px solid rgba(255,255,255,0.08); padding: 20px; border-radius: 12px; height: 260px; text-align: center;'>
            <h4 style='color:#ffb86c; margin-top:0;'>SCORE 2: YoY MOMENTUM</h4>
            <p style='font-size: 0.85rem; color:#9ca3af;'>Velocity & Trajectory Change</p>
            <div style='display: flex; justify-content: space-around; margin-top:20px;'>
                <div>
                    <span style='font-size:0.8rem; color:#8be9fd;'>Speed (16D Norm)</span>
                    <div style='font-size: 1.8rem; font-weight:800; color:#ffb86c;'>{s2_vel:.4f}</div>
                </div>
                <div>
                    <span style='font-size:0.8rem; color:#ff79c6;'>Direction (Trend)</span>
                    <div style='font-size: 1.8rem; font-weight:800; color:{"#10b981" if s2_dir > 0.1 else "#ef4444" if s2_dir < -0.1 else "#9ca3af"};'>{s2_dir:+.4f}</div>
                </div>
            </div>
            <div style='margin-top: 15px; font-size: 0.85rem;'>
                <span>Momentum Status: </span>
                <b>{"🟢 IMPROVING" if s2_dir > 0.1 else "🔴 DECAYING" if s2_dir < -0.1 else "⚪ STABLE"}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_s3:
        st.markdown(f"""
        <div style='background-color:#0b0c10; border: 1px solid rgba(255,255,255,0.08); padding: 20px; border-radius: 12px; height: 260px; text-align: center;'>
            <h4 style='color:#8be9fd; margin-top:0;'>SCORE 3: DECOUPLING</h4>
            <p style='font-size: 0.85rem; color:#9ca3af;'>Multi-Modal Vector Alignment</p>
            <div style='display: flex; justify-content: space-around; margin-top:20px;'>
                <div>
                    <span style='font-size:0.8rem; color:#8be9fd;'>Latent Gap</span>
                    <div style='font-size: 1.8rem; font-weight:800; color:#bd93f9;'>{s3_gap:.4f}</div>
                </div>
                <div>
                    <span style='font-size:0.8rem; color:#ff79c6;'>Signed Div.</span>
                    <div style='font-size: 1.8rem; font-weight:800; color:{"#ef4444" if abs(s3_div) > 0.3 else "#f59e0b" if abs(s3_div) > 0.15 else "#10b981"};'>{s3_div:+.4f}</div>
                </div>
            </div>
            <div style='margin-top: 15px; font-size: 0.85rem;'>
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
                    <h4 style='color:#bd93f9; margin-bottom: 5px; margin-top: 0;'>Dimension {dim_num}</h4>
                    <p style='font-size: 1.8rem; font-weight: 700; color:{card_color}; margin: 0;'>{val:.4f}</p>
                    <p style='font-size: 0.85rem; color:#8be9fd; margin: 0;'>Active Driver</p>
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

# --- ARCHITECTURE DETAILS FOOTER ---
st.markdown("---")
st.subheader("🛡️ Aegis Unsupervised Dual-Tower Architecture")
st.write("""
This model is trained in two phases to guarantee extreme robustness without relying on synthetic defaults:
1. **The Financial Autoencoder:** Ingests 41 financial ratios over a rolling 3-year window and compresses them to a 16-dimensional coordinate vector. It maps healthy stable firms into a dense cluster (**Continent of Normalcy**) and distressed firms onto outlier edges (**Cliff Edge**).
2. **Contrastive LSTM Alignment:** A parallel Recurrent Neural Network (LSTM) ingests textual disclosures. It is trained via **Contrastive Alignment and Negative Sampling** to map the qualitative text vector onto the *exact same 16D coordinates* as the financial vector, using the financial space as a geometric Rosetta Stone.
""")
