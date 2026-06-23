# Aegis 🛡️ 

### Next-Generation AI Corporate Financial Distress Evaluation Engine & Hyperspace Risk Map

Aegis is a multi-modal deep learning framework designed to predict corporate financial distress and default. While traditional credit risk models (like the Altman Z-score) depend on rigid, static, linear combinations of standard accounting ratios, Aegis leverages a dynamic, non-linear representation of corporate health.

---

## 🌟 Core Architectural Innovations

1. **Unsupervised Latent Topology (Phase 1):** 
   A continuous 16-dimensional risk hyperspace trained using an unsupervised deep PyTorch autoencoder. This network discovers underlying non-linear correlations across 41 financial ratios—organicially clustering financially healthy firms in a dense "Continent of Normalcy" and failing ones in a high-risk "Cliff Edge."
   
2. **Self-Supervised Sequential Alignment (Phase 2 / Method A):** 
   Dual-tower Long Short-Term Memory (LSTM) networks trained via contrastive self-supervised learning. This aligns 3-year sequential financial ratio trajectories with dense semantic vectors extracted from qualitative 10-K Management's Discussion and Analysis (MD&A) filings into the exact same 16D space.
   
3. **Multi-Modal Decoupling Discovery:** 
   Aegis monitors **YoY Divergence** (the physical distance/decoupling between a company's financial numbers and its management's textual narrative). Sudden drops in narrative-to-financial alignment serve as powerful early-warning signals up to 12 months ahead of traditional credit agency downgrades.
   
4. **Explainable AI (XAI) Latent Interpreter:** 
   Dissects the coordinates of the 16 joint latent dimensions to highlight the exact financial ratios and qualitative executive discussion themes driving a company's position on the risk map.

---

## 🚀 Live Demo App

This repository houses a clean, relative-path-friendly, and lightweight **Streamlit Interactive App** designed to showcase Aegis to the public and potential clients.

### Features:
* **Interactive 16D Risk Space Projection:** Select from 80,000+ historical corporate snapshots and watch their trajectories mapped in real-time.
* **LinkedIn Demo Case Studies:** Jump straight to curated high-profile crisis cases (e.g., *Party City's* pandemic decoupling, *Rite Aid's* chronic opioid liability stress, and *Catalent's* emerging Wegovy supply-chain bottlenecks).
* **Private Firm Evaluator:** Input a private company's fundamentals manually or via sliders to compute its coordinates and distress scores on the fly.
* **XAI Interpreter:** Drill down into exactly which balance sheet ratios or narrative themes are steering a company's distress coordinates.

---

## 📦 Running Locally

To run the Streamlit dashboard on your local machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/marcionegrao/aegis-risk-evaluator.git
   cd aegis-risk-evaluator
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit server:**
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 🧬 Model Training Assets (Included)
* `primary_encoder_5yr.pth`: PyTorch autoencoder weights for Phase 1.
* `risk_classifier_5yr.pkl`: Scikit-learn random forest classifier representing the distress probability boundary.
* `xai_correlations.pkl`: Cached correlation matrix mapping the 16 latent coordinates to qualitative and quantitative metrics.
* `*_5yr.csv` & `seq_latent_coordinates.csv`: Pre-calculated embeddings for instant visualization.

---

**Author:** Marcio Negrao  
*Main Project: Risk_Project Corporate Bankruptcy Prediction Model*
