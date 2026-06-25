import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page configuration
st.set_page_config(
    page_title="Aegis | AI Corporate Distress Evaluator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-tech financial aesthetic
st.markdown("""
<style>
    /* Premium Pure Black Theme Override */
    .stApp {
        background: #000000 !important;
    }
    .main-header {
        font-size: 3.2rem;
        font-weight: 900;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 0.1rem;
        letter-spacing: 0.08em;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #9CA3AF; /* Muted gray subhead */
        text-align: center;
        margin-bottom: 2.5rem;
        letter-spacing: 0.04em;
    }
    .metric-card {
        background-color: #0B0C10; /* Deep Obsidian card */
        border: 1px solid rgba(255, 255, 255, 0.08); /* Thin elegant border */
        padding: 22px;
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
</style>
""", unsafe_allow_html=True)

# --- 1. DEFINE ARCHITECTURES ---
class RiskAutoencoder(nn.Module):
    def __init__(self, input_dim=41, bottleneck_dim=16):
        super(RiskAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# --- 2. CACHED DATA LOADING & SETUP ---
@st.cache_resource
def load_models_and_scalers():
    # Detect the directory where this script is located for relative paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load PyTorch model weights
    model = RiskAutoencoder(input_dim=41, bottleneck_dim=16)
    model.load_state_dict(torch.load(os.path.join(base_dir, "primary_encoder_5yr.pth"), map_location=torch.device('cpu')))
    model.eval()
    
    # Load Classifier
    clf = joblib.load(os.path.join(base_dir, "risk_classifier_5yr.pkl"))
    
    # Load Master Financials and Fit Scaler (so scaling is 100% accurate)
    df_ratios = pd.read_csv(os.path.join(base_dir, "calculated_financial_ratios_5yr.csv"))
    features = df_ratios.drop(columns=['CIK', 'Company_Name', 'Date']).fillna(0)
    
    scaler = StandardScaler()
    scaler.fit(features)
    
    # Load pre-calculated coordinates
    df_coords = pd.read_csv(os.path.join(base_dir, "latent_coordinates_5yr.csv"))
    df_coords['CIK_int'] = df_coords['CIK'].astype(int)
    
    # Fit PCA for squashing 16D -> 2D
    z_cols = [f'Z{i+1}' for i in range(16)]
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(df_coords[z_cols])
    df_coords['PCA1'] = X_2d[:, 0]
    df_coords['PCA2'] = X_2d[:, 1]
    
    # Bankrupt list
    distressed_ciks = [886158, 1592866, 84129, 1813756, 1483510, 77149, 895126, 86874, 1005414, 13619, 13063, 20171]
    df_coords['Status'] = df_coords['CIK_int'].apply(lambda x: 'Bankrupt' if x in distressed_ciks else 'Healthy')
    
    # Load final risk scores for quick lookups
    df_scores = pd.read_csv(os.path.join(base_dir, "final_risk_scores_5yr.csv"))
    
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
    
    df_seq['Status'] = df_seq['CIK_int'].apply(lambda x: 'Bankrupt' if x in distressed_ciks else 'Healthy')
    
    # Load Aegis Three Scores
    df_aegis = pd.read_csv(os.path.join(base_dir, "aegis_three_scores.csv"))
    df_aegis['CIK_int'] = df_aegis['CIK'].astype(int)
    
    return model, clf, scaler, pca, df_coords, df_scores, distressed_ciks, z_cols, df_seq, pca_seq, df_aegis

# Load resources
try:
    model, clf, scaler, pca, df_coords, df_scores, distressed_ciks, z_cols, df_seq, pca_seq, df_aegis = load_models_and_scalers()
except Exception as e:
    st.error(f"Error loading models or files: {str(e)}")
    st.stop()

# Title banner
st.markdown("<div class='main-header'>AEGIS 🛡️</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Corporate Distress Evaluation Engine & Hyperspace Risk Map</div>", unsafe_allow_html=True)

# Sidebar Page Selection
page = st.sidebar.radio("Navigation", [
    "1. Explore Corporate Risk Space", 
    "2. Sequential Multi-Modal Space (Dual-Tower)", 
    "3. Private Firm Risk Evaluator", 
    "4. Model Architecture & Theory"
])

if page == "1. Explore Corporate Risk Space":
    st.header("1. Explore Corporate Risk Space (80,000+ Snapshots)")
    st.write("Browse and select any public company to project its financial trajectory onto our 16-dimensional risk map.")

    # LinkedIn Demo Case Studies Quick-Select
    demo_cases = {
        "Select a Case Study...": None,
        "Party City (The Classic Pandemic Decoupling)": "PARTY CITY HOLDCO INC.",
        "Rite Aid Corp (Chronic Opioid Liability Decay)": "RITE AID CORP",
        "Catalent, Inc. (The Wegovy Syringe Factory Crisis)": "CATALENT, INC.",
        "Chesapeake Energy (The Pure Balance-Sheet Collapse)": "CHESAPEAKE ENERGY CORP",
        "NVIDIA Corp (The Sovereign Giant of Normalcy)": "NVIDIA CORP"
    }
    
    st.markdown("💡 **LinkedIn Demo Case Studies:** Select a high-profile case study below to instantly see how Aegis performs on historic default and crisis trajectories.")
    selected_demo = st.selectbox("Choose a Case Study to Load", list(demo_cases.keys()))
    
    # Select Box for Companies
    companies_list = df_coords[['CIK_int', 'Company_Name']].drop_duplicates().sort_values('Company_Name')
    
    # Set default company based on selected demo
    default_company = "NVIDIA CORP"
    if selected_demo and demo_cases[selected_demo]:
        default_company = demo_cases[selected_demo]
        
    try:
        default_idx = int(companies_list['Company_Name'].tolist().index(default_company))
    except Exception:
        default_idx = 0
        
    selected_company = st.selectbox("Search / Select a Public Company", companies_list['Company_Name'], index=default_idx)
    selected_cik = companies_list[companies_list['Company_Name'] == selected_company]['CIK_int'].values[0]

    # Subsample data for Plotly snappy rendering (e.g., sample 3000 healthy, but keep ALL bankrupts)
    healthy_coords = df_coords[df_coords['Status'] == 'Healthy']
    bankrupt_coords = df_coords[df_coords['Status'] == 'Bankrupt']
    
    # Take a representative sample of healthy snapshots
    healthy_sample = healthy_coords.sample(n=min(1500, len(healthy_coords)), random_state=42)
    plot_df = pd.concat([healthy_sample, bankrupt_coords], ignore_index=True)
    
    # Add our target company's historical snapshots (guarantees they show up)
    target_snapshots = df_coords[df_coords['CIK_int'] == selected_cik].copy()
    plot_df = pd.concat([plot_df, target_snapshots], ignore_index=True).drop_duplicates(subset=['CIK', 'Date'])
    
    # Map colors
    plot_df['Display_Status'] = plot_df['Status']
    # Mark the specific target company
    plot_df.loc[plot_df['CIK_int'] == selected_cik, 'Display_Status'] = 'Selected Company (History)'

    # Generate 2D Scatter Plot
    fig = px.scatter(
        plot_df, x='PCA1', y='PCA2',
        color='Display_Status',
        hover_data=['Company_Name', 'Date'],
        color_discrete_map={
            'Healthy': 'rgba(31, 119, 180, 0.15)', # Soft, highly transparent blue
            'Bankrupt': 'rgba(214, 39, 40, 0.85)', # Vibrant red
            'Selected Company (History)': '#ffb86c' # Glowing gold
        },
        title=f"2D Projection: {selected_company} Trajectory relative to distress boundary",
        labels={'PCA1': 'Primary Financial Variance (PCA1)', 'PCA2': 'Secondary Financial Variance (PCA2)'},
        render_mode='webgl'
    )
    
    # Customize layout
    fig.update_layout(
        template="plotly_dark",
        height=650,
        legend_title="Company Status",
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Draw trajectory line for the selected company
    if len(target_snapshots) > 1:
        target_snapshots = target_snapshots.sort_values('Date')
        fig.add_trace(go.Scatter(
            x=target_snapshots['PCA1'],
            y=target_snapshots['PCA2'],
            mode='lines+markers',
            name='Historical Path',
            line=dict(color='#ffb86c', width=3, dash='solid'),
            marker=dict(color='#ff79c6', size=8)
        ))

    # Display plot
    st.plotly_chart(fig, use_container_width=True)

    # Display Company Historical Stats and Risk Scores
    st.subheader(f"🛡️ Historical Evaluation Summary: {selected_company}")
    
    # Filter scores
    company_scores = df_scores[df_scores['CIK'] == selected_cik].sort_values('Date')
    
    # Format Date column for display (e.g. 20211231 -> 2021-12-31)
    def format_date(d):
        d_str = str(int(d))
        if len(d_str) == 8:
            return f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:]}"
        return d_str
        
    company_scores['Filing_Date'] = company_scores['Date'].apply(format_date)
    
    cols = st.columns(len(company_scores[-5:])) # Show last 5 reports
    for i, row in enumerate(company_scores[-5:].itertuples()):
        with cols[i]:
            score = row.Risk_Score_100
            if score >= 70:
                color_class = "metric-value-healthy"
                status_text = "🟢 HEALTHY"
            elif score >= 40:
                color_class = "metric-value-caution" # Premium Amber Caution
                status_text = "🟡 CAUTION"
            else:
                color_class = "metric-value-distressed"
                status_text = "🚨 DISTRESSED"
                
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Filing Date:<br><b>{row.Filing_Date}</b></h4>
                <div class='{color_class}'>{score}/100</div>
                <p><b>{status_text}</b></p>
            </div>
            """, unsafe_allow_html=True)


elif page == "2. Sequential Multi-Modal Space (Dual-Tower)":
    st.header("🔗 2. Sequential Multi-Modal Space (Dual-Tower Alignment)")
    st.write("""
    This map displays corporate trajectories through our unified 16-dimensional latent space. 
    Using contrastive self-supervised alignment (**Method A**), we mapped sequential 3-year financial ratios and corresponding 10-K MD&A textual narratives into the *exact same space*. 
    
    When a company's financial health shifts, we can observe the path of its numbers (**Financial Trajectory**) and the path of its management's story (**Textual Narrative Trajectory**) over time.
    """)

    # LinkedIn Demo Case Studies Quick-Select
    demo_cases_seq = {
        "Select a Case Study...": None,
        "Party City (The Classic Pandemic Decoupling)": "PARTY CITY HOLDCO INC.",
        "Rite Aid Corp (Chronic Opioid Liability Decay)": "RITE AID CORP",
        "Catalent, Inc. (The Wegovy Syringe Factory Crisis)": "CATALENT, INC.",
        "Chesapeake Energy (The Pure Balance-Sheet Collapse)": "CHESAPEAKE ENERGY CORP",
        "NVIDIA Corp (The Sovereign Giant of Normalcy)": "NVIDIA CORP"
    }
    
    st.markdown("💡 **LinkedIn Demo Case Studies:** Select a high-profile case study below to see how executive narrative (10-K text) decoupled from actual financials.")
    selected_seq_demo = st.selectbox("Choose a Case Study to Load", list(demo_cases_seq.keys()))
    
    # Select Box for Companies in sequential dataset
    seq_companies = df_seq[['CIK_int', 'Company_Name']].drop_duplicates().sort_values('Company_Name')
    
    # Set default company based on selected demo
    default_seq_company = "NVIDIA CORP"
    if selected_seq_demo and demo_cases_seq[selected_seq_demo]:
        default_seq_company = demo_cases_seq[selected_seq_demo]
        
    try:
        default_seq_idx = int(seq_companies['Company_Name'].tolist().index(default_seq_company))
    except Exception:
        default_seq_idx = 0
        
    selected_seq_company = st.selectbox("Search / Select a Company in Sequential Space", seq_companies['Company_Name'], index=default_seq_idx)
    selected_seq_cik = seq_companies[seq_companies['Company_Name'] == selected_seq_company]['CIK_int'].values[0]

    # Subsample background points for quick rendering
    healthy_seq = df_seq[df_seq['Status'] == 'Healthy']
    bankrupt_seq = df_seq[df_seq['Status'] == 'Bankrupt']
    
    healthy_seq_sample = healthy_seq.sample(n=min(1000, len(healthy_seq)), random_state=42)
    plot_seq_df = pd.concat([healthy_seq_sample, bankrupt_seq], ignore_index=True)
    
    # Isolate target company sequential history
    target_seq = df_seq[df_seq['CIK_int'] == selected_seq_cik].copy().sort_values('Filing_Year_End')
    
    # We want to show background points
    # Color background points
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
        title=f"Dual-Trajectory Projection: {selected_seq_company} in Aligned Latent Space",
        labels={'PCA_F1': 'Aligned Component 1', 'PCA_F2': 'Aligned Component 2'},
        render_mode='webgl'
    )
    
    fig_seq.update_layout(
        template="plotly_dark",
        height=650,
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Plot paths if we have records
    if len(target_seq) > 0:
        # 1. Add Financial Path
        fig_seq.add_trace(go.Scatter(
            x=target_seq['PCA_F1'],
            y=target_seq['PCA_F2'],
            mode='lines+markers+text',
            name='Financial Trajectory (ZF)',
            text=target_seq['Filing_Year_End'].astype(str),
            textposition="top center",
            line=dict(color='#ffb86c', width=4),
            marker=dict(color='#ff79c6', size=10, symbol='circle')
        ))
        
        # 2. Add Textual Path
        fig_seq.add_trace(go.Scatter(
            x=target_seq['PCA_T1'],
            y=target_seq['PCA_T2'],
            mode='lines+markers+text',
            name='Narrative Trajectory (ZT)',
            text=target_seq['Filing_Year_End'].astype(str) + " Text",
            textposition="bottom center",
            line=dict(color='#8be9fd', width=3, dash='dash'),
            marker=dict(color='#bd93f9', size=10, symbol='diamond')
        ))
        
    st.plotly_chart(fig_seq, use_container_width=True)

    # ---------------------------------------------------------
    # DISPLAY METRICS AND THREE-SCORE ANALYSIS
    # ---------------------------------------------------------
    st.subheader(f"📊 Aegis Three-Score Diagnostics: {selected_seq_company}")
    
    # Filter the Aegis Scores for our selected company
    company_aegis = df_aegis[df_aegis['CIK_int'] == selected_seq_cik].sort_values('Filing_Year_End')
    
    if len(company_aegis) > 0:
        # We can display a high-tech dashboard overview table
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
        
        st.markdown("### 🔬 Year-Specific Deep Dive")
        selected_year = st.selectbox("Select Diagnostic Year", company_aegis['Filing_Year_End'].tolist())
        
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
            
        # Sophisticated multi-dimensional interpretation of decoupling
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
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        xai_data = joblib.load(os.path.join(base_dir, "xai_correlations.pkl"))
    except Exception as e:
        xai_data = None
        
    if xai_data is not None:
        st.markdown("---")
        st.subheader("🔍 Aegis Explainable AI (XAI) Latent Interpreter")
        st.write("""
        Aegis maps financial ratios and qualitative narratives into 16 joint latent dimensions. 
        Select a year to dissect **exactly** which financial ratios and narrative concepts are driving the company's coordinates.
        """)
        
        selected_year = st.selectbox("Select Year to Explain", target_seq['Filing_Year_End'].tolist())
        
        # Extract coordinates for the selected year
        year_rows = target_seq[target_seq['Filing_Year_End'] == selected_year]
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
                    card_color = "#ff5555" if val < -0.2 else "#50fa7b" if val > 0.2 else "#f1fa8c"
                    st.markdown(f"""
                    <div style='background-color:#1e1f29; border: 1px solid #44475a; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 10px;'>
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
                        
                    st.markdown("**Associated Narrative Themes:**")
                    for t_name, t_val in top_themes.items():
                        top_words = xai_data['theme_words'][t_name][:5]
                        st.markdown(f"- **{t_name.replace('_', ' ')}** ({t_val:+.2f})")
                        st.caption(f"  *Key words:* {', '.join(top_words)}")


elif page == "3. Private Firm Risk Evaluator":
    st.header("🛡️ Private Firm Financial Distress Evaluator")
    st.write("Input a private company's fundamental financial statements. Aegis will compute 41 ratios on the fly, map them through our 16D encoder, and project them on the global map!")

    # Tab selector for Manual Input or CSV Template Upload
    input_mode = st.radio("Choose Input Method", ["Manual Statement Input", "Slider-Based Direct Ratio Input"])

    ratios_dict = {}

    if input_mode == "Manual Statement Input":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Balance Sheet")
            cash = st.number_input("Cash & Equivalents", value=5000000.0, step=100000.0)
            ar = st.number_input("Accounts Receivable", value=3000000.0, step=100000.0)
            inventory = st.number_input("Inventory", value=4000000.0, step=100000.0)
            current_assets = st.number_input("Total Current Assets", value=15000000.0, step=100000.0)
            total_assets = st.number_input("Total Assets", value=35000000.0, step=500000.0)
            ap = st.number_input("Accounts Payable", value=2000000.0, step=100000.0)
            st_debt = st.number_input("Short Term Debt", value=1000000.0, step=50000.0)
            current_liab = st.number_input("Total Current Liabilities", value=5000000.0, step=100000.0)
            lt_debt = st.number_input("Long Term Debt", value=8000000.0, step=100000.0)
            total_liab = st.number_input("Total Liabilities", value=15000000.0, step=500000.0)
            retained_earnings = st.number_input("Retained Earnings", value=8000000.0, step=200000.0)
            equity = st.number_input("Total Stockholders Equity", value=20000000.0, step=500000.0)
            
        with col2:
            st.subheader("Income Statement")
            revenue = st.number_input("Total Revenue", value=25000000.0, step=500000.0)
            cost_of_rev = st.number_input("Cost of Revenue", value=15000000.0, step=500000.0)
            gross_profit = st.number_input("Gross Profit", value=10000000.0, step=200000.0)
            op_income = st.number_input("Operating Income", value=3000000.0, step=100000.0)
            sga = st.number_input("SG&A Expenses", value=5000000.0, step=100000.0)
            rnd = st.number_input("R&D Expenses", value=1000000.0, step=50000.0)
            op_exp = st.number_input("Total Operating Expenses", value=7000000.0, step=100000.0)
            interest_exp = st.number_input("Interest Expense", value=500000.0, step=20000.0)
            interest_inc = st.number_input("Interest Income", value=50000.0, step=5000.0)
            tax_exp = st.number_input("Income Tax Expense", value=600000.0, step=20000.0)
            net_income = st.number_input("Net Income", value=1950000.0, step=50000.0)

        with col3:
            st.subheader("Cash Flow & CapEx")
            ocf = st.number_input("Operating Cash Flow (OCF)", value=4000000.0, step=100000.0)
            capex = st.number_input("Capital Expenditures (CapEx)", value=2000000.0, step=100000.0)
            depr = st.number_input("Depreciation & Amortization", value=1200000.0, step=50000.0)
            
            st.subheader("Evaluate private company")
            evaluate_btn = st.button("RUN DISTRESS INFERENCE 🛡️", use_container_width=True)
            
        # Calculation Logic
        total_debt = st_debt + lt_debt
        ebit = net_income + interest_exp + tax_exp
        pretax_income = net_income + tax_exp
        fcf = ocf - capex
        invested_capital = total_debt + equity
        working_capital = current_assets - current_liab

        # Safely calculate the 41 financial ratios
        def safe_div(n, d):
            return n / d if d != 0 else 0.0

        ratios_dict['Current_Ratio'] = safe_div(current_assets, current_liab)
        ratios_dict['Quick_Ratio'] = safe_div(cash + ar, current_liab)
        ratios_dict['Cash_Ratio'] = safe_div(cash, current_liab)
        ratios_dict['OCF_to_Current_Liabilities'] = safe_div(ocf, current_liab)
        ratios_dict['Working_Capital_to_Assets'] = safe_div(working_capital, total_assets)
        ratios_dict['Debt_to_Equity'] = safe_div(total_debt, equity)
        ratios_dict['Debt_to_Assets'] = safe_div(total_debt, total_assets)
        ratios_dict['LongTermDebt_to_Assets'] = safe_div(lt_debt, total_assets)
        ratios_dict['ShortTermDebt_to_TotalDebt'] = safe_div(st_debt, total_debt)
        ratios_dict['Equity_to_Liabilities'] = safe_div(equity, total_liab)
        ratios_dict['Interest_Coverage_EBIT'] = safe_div(ebit, interest_exp)
        ratios_dict['Interest_Coverage_OCF'] = safe_div(ocf, interest_exp)
        ratios_dict['Cash_to_TotalDebt'] = safe_div(cash, total_debt)
        ratios_dict['OCF_to_TotalDebt'] = safe_div(ocf, total_debt)
        ratios_dict['FCF_to_TotalDebt'] = safe_div(fcf, total_debt)
        ratios_dict['Gross_Margin'] = safe_div(gross_profit, revenue)
        ratios_dict['Operating_Margin'] = safe_div(op_income, revenue)
        ratios_dict['Net_Margin'] = safe_div(net_income, revenue)
        ratios_dict['PreTax_Margin'] = safe_div(pretax_income, revenue)
        ratios_dict['EBIT_Margin'] = safe_div(ebit, revenue)
        ratios_dict['ROA'] = safe_div(net_income, total_assets)
        ratios_dict['ROE'] = safe_div(net_income, equity)
        ratios_dict['ROIC'] = safe_div(ebit, invested_capital)
        ratios_dict['RetainedEarnings_to_Assets'] = safe_div(retained_earnings, total_assets)
        ratios_dict['EBIT_to_Assets'] = safe_div(ebit, total_assets)
        ratios_dict['Asset_Turnover'] = safe_div(revenue, total_assets)
        ratios_dict['Inventory_Turnover'] = safe_div(cost_of_rev, inventory)
        ratios_dict['Receivables_Turnover'] = safe_div(revenue, ar)
        ratios_dict['Payables_Turnover'] = safe_div(cost_of_rev, ap)
        ratios_dict['Fixed_Asset_Turnover'] = safe_div(revenue, total_assets - current_assets)
        ratios_dict['SGA_to_Revenue'] = safe_div(sga, revenue)
        ratios_dict['RnD_to_Revenue'] = safe_div(rnd, revenue)
        ratios_dict['OperatingExpenses_to_Revenue'] = safe_div(op_exp, revenue)
        ratios_dict['OCF_to_Revenue'] = safe_div(ocf, revenue)
        ratios_dict['FCF_to_Revenue'] = safe_div(fcf, revenue)
        ratios_dict['OCF_to_NetIncome'] = safe_div(ocf, net_income)
        ratios_dict['CapEx_to_Revenue'] = safe_div(capex, revenue)
        ratios_dict['CapEx_to_OCF'] = safe_div(capex, ocf)
        ratios_dict['Depreciation_to_CapEx'] = safe_div(depr, capex)
        ratios_dict['Effective_Tax_Rate'] = safe_div(tax_exp, pretax_income)
        ratios_dict['InterestIncome_to_InterestExpense'] = safe_div(interest_inc, interest_exp)

    elif input_mode == "Slider-Based Direct Ratio Input":
        st.write("Adjust the most significant financial ratio values directly below:")
        col1, col2 = st.columns(2)
        with col1:
            ratios_dict['Current_Ratio'] = st.slider("Current Ratio", 0.0, 5.0, 1.8)
            ratios_dict['Quick_Ratio'] = st.slider("Quick Ratio", 0.0, 4.0, 1.2)
            ratios_dict['Working_Capital_to_Assets'] = st.slider("Working Capital to Assets (Z-Score essential)", -1.0, 1.0, 0.2)
            ratios_dict['Debt_to_Assets'] = st.slider("Debt to Assets", 0.0, 1.5, 0.4)
            ratios_dict['Debt_to_Equity'] = st.slider("Debt to Equity", 0.0, 5.0, 1.1)
            ratios_dict['RetainedEarnings_to_Assets'] = st.slider("Retained Earnings to Assets", -1.0, 1.0, 0.15)
            ratios_dict['ROA'] = st.slider("Return on Assets (ROA)", -0.5, 0.5, 0.06)
        with col2:
            ratios_dict['Operating_Margin'] = st.slider("Operating Margin", -1.0, 1.0, 0.12)
            ratios_dict['Net_Margin'] = st.slider("Net Margin", -1.0, 1.0, 0.08)
            ratios_dict['Asset_Turnover'] = st.slider("Asset Turnover Ratio", 0.0, 4.0, 1.0)
            ratios_dict['OCF_to_TotalDebt'] = st.slider("Operating Cash Flow to Total Debt", -1.0, 1.0, 0.3)
            ratios_dict['FCF_to_TotalDebt'] = st.slider("Free Cash Flow to Total Debt", -1.0, 1.0, 0.15)
            ratios_dict['Interest_Coverage_EBIT'] = st.slider("Interest Coverage Ratio (EBIT)", -10.0, 50.0, 6.0)
            ratios_dict['SGA_to_Revenue'] = st.slider("SG&A to Revenue Ratio", 0.0, 1.0, 0.25)

        # Fill any missing ratios with defaults
        all_cols = df_coords.columns[3:44] # Get the 41 original ratios
        for col in all_cols:
            if col not in ratios_dict:
                ratios_dict[col] = 0.0
                
        evaluate_btn = st.button("RUN DISTRESS INFERENCE 🛡️", use_container_width=True)

    if evaluate_btn:
        # Convert dictionary to DataFrame aligned with our exact 41 columns
        input_cols = list(scaler.feature_names_in_)
        input_data = pd.DataFrame([[ratios_dict.get(c, 0.0) for c in input_cols]], columns=input_cols)
        
        # Scale inputs using fit scaler
        scaled_input = scaler.transform(input_data)
        X_test_tensor = torch.FloatTensor(scaled_input)
        
        # Pass through primary encoder to get 16D coordinates
        with torch.no_grad():
            reconstructed, latent_coords = model(X_test_tensor)
            
        # Get risk score from classifier
        prob_healthy = clf.predict_proba(latent_coords.numpy())[:, 0][0]
        final_score = round(prob_healthy * 100, 1)
        
        # Project 16D -> 2D using our PCA
        pca_coords = pca.transform(latent_coords.numpy())[0]
        
        # Calculate reconstruction error (Anomalous Signature)
        mse_loss = nn.MSELoss()(reconstructed, X_test_tensor).item()
        
        st.write("---")
        st.subheader("🎯 Evaluation Results")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            if final_score >= 70:
                score_col = "metric-value-healthy"
                status = "🟢 HEALTHY"
            elif final_score >= 40:
                score_col = "metric-value-caution" # Premium Amber Caution
                status = "🟡 CAUTION"
            else:
                score_col = "metric-value-distressed"
                status = "🚨 CRITICAL DISTRESS"
                
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Financial Distress Score</h4>
                <div class='{score_col}'>{final_score}/100</div>
                <p><b>Status: {status}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Reconstruction Loss (Anomalous)</h4>
                <div class='metric-value-healthy' style='color:#bd93f9'>{mse_loss:.4f}</div>
                <p><b>Low error = standard firm signature</b></p>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            # Short advice
            if final_score >= 70:
                desc = "The company displays strong liquidity, robust margins, and safe leverage metrics. Negligible distress risk."
            elif final_score >= 40:
                desc = "Noticeable friction in cash flows or slightly elevated leverage. The firm is stable but shows early signals of concern."
            else:
                desc = "Urgent attention required. The firm exhibits critical distress signatures identical to historic bankruptcies (severe debt, negative ROA)."
            st.markdown(f"""
            <div class='metric-card' style='text-align: left;'>
                <h4>Aegis Diagnostic Advice</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

        # Draw the target firm on the interactive map
        # Sample healthy background
        healthy_coords = df_coords[df_coords['Status'] == 'Healthy']
        bankrupt_coords = df_coords[df_coords['Status'] == 'Bankrupt']
        healthy_sample = healthy_coords.sample(n=1500, random_state=42)
        plot_df = pd.concat([healthy_sample, bankrupt_coords], ignore_index=True)
        
        # Insert target private firm
        private_row = pd.DataFrame([{
            'Company_Name': 'TARGET PRIVATE FIRM',
            'Date': 'Evaluation Day',
            'PCA1': pca_coords[0],
            'PCA2': pca_coords[1],
            'Status': 'TARGET PRIVATE FIRM'
        }])
        plot_df = pd.concat([plot_df, private_row], ignore_index=True)

        # Plotly graph
        fig_private = px.scatter(
            plot_df, x='PCA1', y='PCA2',
            color='Status',
            hover_data=['Company_Name', 'Date'],
            color_discrete_map={
                'Healthy': 'rgba(31, 119, 180, 0.1)', # highly transparent blue
                'Bankrupt': 'rgba(214, 39, 40, 0.8)', # red
                'TARGET PRIVATE FIRM': '#ff5555' if final_score < 40 else '#50fa7b' # Green if safe, red if distressed
            },
            title="Interactive Map: Target Private Company plotted on Global Distress Topology",
            labels={'PCA1': 'PCA1', 'PCA2': 'PCA2'},
            render_mode='webgl'
        )
        
        # Add a custom golden star marker for the target firm to make it pop!
        fig_private.add_trace(go.Scatter(
            x=[pca_coords[0]],
            y=[pca_coords[1]],
            mode='markers',
            name='EVALUATED FIRM',
            marker=dict(color='#ffb86c', size=18, symbol='star', line=dict(color='white', width=1.5))
        ))
        
        fig_private.update_layout(
            template="plotly_dark",
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_private, use_container_width=True)


elif page == "4. Model Architecture & Theory":
    st.header("🛡️ Aegis Design Philosophy & Math")
    st.write("""
    ### Why traditional scoring models (Altman Z-Score) fail
    Standard credit scoring models depend heavily on rigid, static linear regressions created decades ago (e.g. Altman's 1968 coefficients). They fail to capture modern corporate complexities, complex non-linear balance sheet structures, and sector-specific asset valuations.
    
    ### Aegis: The 16-Dimensional Neural Alternative
    Instead of hardcoding a scoring formula, Aegis learns the **underlying structure of corporate finance** using an unsupervised deep PyTorch autoencoder.
    
    1. **The Primary Encoder (10-K Ratios ➔ 16D Latent Coordinates):** 
       We input 41 financial ratios into a feedforward neural network that compresses them down into a 16-dimensional "hyperspace." The network is trained to minimize *reconstruction error*—meaning it seeks the absolute best latent representation to understand a company's state.
    
    2. **Visualizing the Hyperspace (PCA / UMAP):** 
       By applying dimensionality reduction (like Principal Component Analysis) on the 16D coordinate vector, we map any company's snapshot onto a 2D canvas. Safe, highly liquid companies organically cluster together in the **"Continent of Normalcy"** (safe zone), while highly leveraged, failing companies naturally drift toward the **"Cliff Edge"** (distress zone).
       
    3. **Boundary Classification (Random Forest Risk Boundary):** 
       A secondary classification engine is trained on the 16D coordinate mappings of known bankruptcies (e.g., Bed Bath & Beyond, Party City) and healthy giants to construct a high-dimensional probability boundary. This boundary output generates our continuous **0 - 100 Financial Distress Score**.
    """)
