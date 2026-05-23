"""
Streamlit app — E-commerce Customer Segmentation
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load model + data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    kmeans = joblib.load(os.path.join(SAVE_DIR, 'kmeans_model.pkl'))
    scaler = joblib.load(os.path.join(SAVE_DIR, 'scaler.pkl'))
    return kmeans, scaler

@st.cache_data
def load_data():
    rfm      = pd.read_csv(os.path.join(SAVE_DIR, 'rfm_customers.csv'))
    profiles = pd.read_csv(os.path.join(SAVE_DIR, 'cluster_profiles.csv'))
    return rfm, profiles

# ── Build dynamic segment descriptions from cluster_profiles.csv ──────────────
def build_segment_map(profiles: pd.DataFrame) -> dict:
    """
    Returns {cluster_id: {"name": ..., "color": ..., "desc": ...}}
    """
    palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
    desc_map = {
        "Champions":         "Bought recently, buy often and spend the most.",
        "Loyal Customers":   "Buy regularly with good monetary value.",
        "Potential Loyalists": "Recent customers with moderate frequency — room to grow.",
        "At Risk":           "Previously active buyers who haven't purchased recently.",
        "At Risk (C3)":      "Lapsed customers with low frequency and low spend.",
        "At Risk (C2)":      "At-risk customers with moderate past activity.",
    }
    seg_map = {}
    for _, row in profiles.iterrows():
        cid  = int(row['Cluster'])
        seg  = str(row['Segment'])
        color = palette[cid % len(palette)]
        description = desc_map.get(seg, f"Customer group {cid}")
        seg_map[cid] = {"name": seg, "color": color, "desc": description}
    return seg_map

# ── Marketing recommendations ─────────────────────────────────────────────────
RECOMMENDATIONS = {
    "Champions": [
        "Reward with exclusive loyalty programme benefits",
        "Offer early access to new products and sales",
        "Request product reviews and referrals",
        "Upsell premium and complementary products",
    ],
    "Loyal Customers": [
        "Enroll in points-based loyalty scheme",
        "Send personalised product recommendations",
        "Offer subscription deals to maintain frequency",
        "Cross-sell related product categories",
    ],
    "Potential Loyalists": [
        "Send welcome series with personalised onboarding",
        "Offer first repeat-purchase discount",
        "Highlight best-sellers and trending items",
        "Invite to loyalty programme at lower threshold",
    ],
    "At Risk": [
        "Win-back campaign with time-limited discount",
        "'We miss you' personalised email series",
        "Survey to understand reasons for disengagement",
        "Showcase new arrivals since last purchase",
    ],
    "At Risk (C3)": [
        "Aggressive win-back offer (20%+ discount)",
        "Last-chance reactivation campaign",
        "Reduce email frequency to avoid unsubscribes",
        "Exit survey for churn insight",
    ],
    "At Risk (C2)": [
        "Targeted reactivation with personalised offers",
        "Share curated recommendations based on past purchases",
        "Bundle deals to increase order value",
        "Loyalty programme invitation",
    ],
}

def get_recommendation(seg_name: str) -> list:
    return RECOMMENDATIONS.get(seg_name, ["Review customer profile and create personalised strategy."])


# ══════════════════════════════════════════════════════════════════════════════
# Load everything
# ══════════════════════════════════════════════════════════════════════════════
try:
    kmeans, scaler = load_model()
    rfm, profiles  = load_data()
    seg_map        = build_segment_map(profiles)
    model_loaded   = True
except Exception as e:
    model_loaded = False
    load_error   = str(e)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Predict Segment", "About"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model**: K-Means (k=4)")
st.sidebar.markdown("**Features**: RFM (log1p + StandardScaler)")
if model_loaded:
    st.sidebar.success("Model loaded")
else:
    st.sidebar.error("Model not loaded")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("E-commerce Customer Segmentation Dashboard")
    st.markdown("Explore the customer segments identified by K-Means clustering on RFM features.")

    if not model_loaded:
        st.error(f"Could not load model/data: {load_error}")
        st.stop()

    # ── Top KPI metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers",  f"{len(rfm):,}")
    col2.metric("Avg Recency (days)", f"{rfm['Recency'].mean():.0f}")
    col3.metric("Avg Frequency",     f"{rfm['Frequency'].mean():.1f}")
    col4.metric("Avg Monetary (GBP)", f"£{rfm['Monetary'].mean():,.0f}")

    st.markdown("---")

    # ── Segment distribution ──────────────────────────────────────────────────
    st.subheader("Segment Distribution")
    seg_counts = rfm['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    seg_counts['Color'] = seg_counts['Segment'].map(
        {v['name']: v['color'] for v in seg_map.values()}
    )

    col_pie, col_bar = st.columns([1, 1])
    with col_pie:
        fig_pie = px.pie(
            seg_counts,
            names='Segment',
            values='Count',
            color='Segment',
            color_discrete_map={row['Segment']: row['Color'] for _, row in seg_counts.iterrows()},
            title="Customers by Segment",
            hole=0.35,
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_bar_count = px.bar(
            seg_counts.sort_values('Count', ascending=True),
            x='Count',
            y='Segment',
            orientation='h',
            color='Segment',
            color_discrete_map={row['Segment']: row['Color'] for _, row in seg_counts.iterrows()},
            title="Customer Count per Segment",
        )
        fig_bar_count.update_layout(showlegend=False)
        st.plotly_chart(fig_bar_count, use_container_width=True)

    st.markdown("---")

    # ── RFM by segment ────────────────────────────────────────────────────────
    st.subheader("Average RFM Values by Segment")
    seg_rfm = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
    color_map = {v['name']: v['color'] for v in seg_map.values()}

    col_r, col_f, col_m = st.columns(3)
    with col_r:
        fig_r = px.bar(
            seg_rfm.sort_values('Recency'),
            x='Segment', y='Recency',
            color='Segment',
            color_discrete_map=color_map,
            title="Avg Recency (days) — lower is better",
        )
        fig_r.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig_r, use_container_width=True)

    with col_f:
        fig_f = px.bar(
            seg_rfm.sort_values('Frequency', ascending=False),
            x='Segment', y='Frequency',
            color='Segment',
            color_discrete_map=color_map,
            title="Avg Frequency (purchases)",
        )
        fig_f.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig_f, use_container_width=True)

    with col_m:
        fig_m = px.bar(
            seg_rfm.sort_values('Monetary', ascending=False),
            x='Segment', y='Monetary',
            color='Segment',
            color_discrete_map=color_map,
            title="Avg Monetary Value (GBP)",
        )
        fig_m.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("---")

    # ── Cluster profiles table ────────────────────────────────────────────────
    st.subheader("Cluster Profile Summary")
    st.dataframe(
        profiles[['Cluster', 'Segment', 'Count',
                  'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']].rename(columns={
            'Avg_Recency':   'Avg Recency (d)',
            'Avg_Frequency': 'Avg Frequency',
            'Avg_Monetary':  'Avg Monetary (GBP)',
        }),
        use_container_width=True,
    )

    st.markdown("---")

    # ── RFM dataframe with search ─────────────────────────────────────────────
    st.subheader("Customer RFM Data")
    search_seg = st.selectbox(
        "Filter by Segment",
        options=["All"] + sorted(rfm['Segment'].unique().tolist()),
    )
    display_df = rfm if search_seg == "All" else rfm[rfm['Segment'] == search_seg]
    st.dataframe(
        display_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']]
        .sort_values('Monetary', ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=400,
    )
    st.caption(f"Showing {len(display_df):,} customers")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Predict Segment
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict Segment":
    st.title("Predict Customer Segment")
    st.markdown("Enter a customer's RFM values to predict their segment and get a marketing recommendation.")

    if not model_loaded:
        st.error(f"Could not load model/data: {load_error}")
        st.stop()

    # ── Single prediction form ────────────────────────────────────────────────
    st.subheader("Single Customer Prediction")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.slider(
                "Recency (days since last purchase)",
                min_value=1, max_value=400, value=50,
                help="Fewer days = more recent purchase = better"
            )
        with col2:
            frequency = st.slider(
                "Frequency (number of purchases)",
                min_value=1, max_value=100, value=5,
                help="Higher = more loyal"
            )
        with col3:
            monetary = st.slider(
                "Monetary (total spend in GBP)",
                min_value=1, max_value=50000, value=1000, step=50,
                help="Higher = more valuable customer"
            )
        submit = st.form_submit_button("Predict Segment", type="primary")

    if submit:
        # Transform
        vals_log = np.log1p([[recency, frequency, monetary]])
        vals_scaled = scaler.transform(vals_log)
        cluster_id  = int(kmeans.predict(vals_scaled)[0])
        seg_info    = seg_map.get(cluster_id, {"name": f"Cluster {cluster_id}",
                                               "color": "#888888",
                                               "desc": ""})

        # Display result
        st.markdown("---")
        col_res, col_rec = st.columns([1, 1])

        with col_res:
            st.markdown(
                f"""
                <div style='background-color:{seg_info["color"]}22;
                            border-left: 5px solid {seg_info["color"]};
                            padding: 20px; border-radius: 8px;'>
                    <h2 style='color:{seg_info["color"]};margin:0'>
                        {seg_info["name"]}
                    </h2>
                    <p style='font-size:16px;margin-top:8px'>{seg_info["desc"]}</p>
                    <hr>
                    <b>Input values:</b><br>
                    Recency: <b>{recency} days</b><br>
                    Frequency: <b>{frequency} purchases</b><br>
                    Monetary: <b>£{monetary:,}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Radar / bar showing input vs segment averages
            seg_avg = rfm[rfm['Cluster'] == cluster_id][['Recency', 'Frequency', 'Monetary']].mean()
            comp_df = pd.DataFrame({
                'Metric':    ['Recency', 'Frequency', 'Monetary (÷100)'],
                'Customer':  [recency, frequency, monetary / 100],
                'Segment Avg': [seg_avg['Recency'], seg_avg['Frequency'], seg_avg['Monetary'] / 100],
            })
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(name='This Customer', x=comp_df['Metric'],
                                      y=comp_df['Customer'], marker_color=seg_info['color']))
            fig_comp.add_trace(go.Bar(name='Segment Average', x=comp_df['Metric'],
                                      y=comp_df['Segment Avg'], marker_color='#CCCCCC'))
            fig_comp.update_layout(
                barmode='group',
                title='Customer vs Segment Average',
                height=300,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        with col_rec:
            st.markdown("### Marketing Recommendations")
            recs = get_recommendation(seg_info['name'])
            for rec in recs:
                st.markdown(f"- {rec}")

            st.markdown("### Segment Statistics")
            seg_stats = rfm[rfm['Cluster'] == cluster_id][['Recency', 'Frequency', 'Monetary']].describe().round(1)
            st.dataframe(seg_stats, use_container_width=True)

    st.markdown("---")

    # ── Batch prediction ──────────────────────────────────────────────────────
    st.subheader("Batch Prediction")
    st.markdown(
        "Upload a CSV with columns **Recency**, **Frequency**, **Monetary** "
        "to predict segments for multiple customers at once."
    )

    template_df = pd.DataFrame({
        'CustomerID': [10001, 10002, 10003],
        'Recency':    [15,    120,   250  ],
        'Frequency':  [12,    3,     1    ],
        'Monetary':   [4500,  800,   150  ],
    })
    st.download_button(
        label="Download Template CSV",
        data=template_df.to_csv(index=False).encode('utf-8'),
        file_name='rfm_template.csv',
        mime='text/csv',
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            required = {'Recency', 'Frequency', 'Monetary'}
            if not required.issubset(batch_df.columns):
                st.error(f"CSV must contain columns: {required}")
            else:
                vals_log    = np.log1p(batch_df[['Recency', 'Frequency', 'Monetary']].values)
                vals_scaled = scaler.transform(vals_log)
                batch_df['Cluster'] = kmeans.predict(vals_scaled).astype(int)
                batch_df['Segment'] = batch_df['Cluster'].map(
                    {k: v['name'] for k, v in seg_map.items()}
                )
                st.success(f"Predicted segments for {len(batch_df):,} customers.")
                st.dataframe(batch_df, use_container_width=True)
                st.download_button(
                    label="Download Predictions",
                    data=batch_df.to_csv(index=False).encode('utf-8'),
                    file_name='predictions.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — About
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.title("About This Project")

    st.markdown("""
## E-commerce Customer Segmentation

### Project Overview
This project applies unsupervised machine learning to segment customers of a UK-based
online retailer using the **UCI Online Retail Dataset** (541,909 transactions,
December 2010 – December 2011).

### Methodology

#### 1. Data Preprocessing
- Removed 9,288 cancelled orders and 135,080 rows with missing CustomerIDs
- Filtered out non-positive Quantity and UnitPrice values
- Final clean dataset: **397,884 transactions** across **4,338 customers**

#### 2. RFM Feature Engineering
Customer purchasing behaviour is captured through three metrics:
- **Recency (R)** — days since last purchase
- **Frequency (F)** — number of unique invoices
- **Monetary (M)** — total spend in GBP

#### 3. Feature Scaling
Log1p transformation followed by StandardScaler normalization to handle
right-skewed distributions and ensure equal feature weighting.

#### 4. Clustering Algorithms
""")

    # Algorithm comparison table
    algo_data = {
        'Algorithm':         ['K-Means', 'Hierarchical (Ward)', 'DBSCAN', 'GMM'],
        'Clusters':          ['4', '4', '2 + noise', '4'],
        'Silhouette Score':  ['0.337', '0.242', '0.293', '0.173'],
        'Davies-Bouldin':    ['1.010', '1.120', '1.063', '1.721'],
        'Calinski-Harabasz': ['3329', '2615', '2391', '2151'],
        'Winner':            ['YES', 'No', 'No', 'No'],
    }
    st.dataframe(pd.DataFrame(algo_data), use_container_width=True, hide_index=True)

    st.markdown("""
**K-Means was selected** as the best algorithm based on:
- Highest Silhouette Score (0.337) — best cluster separation
- Lowest Davies-Bouldin Score (1.010) — most compact and well-separated clusters
- Highest Calinski-Harabasz Score (3329) — best ratio of between/within cluster variance

#### 5. Customer Segments
""")

    if model_loaded:
        for _, row in profiles.iterrows():
            cid = int(row['Cluster'])
            seg = str(row['Segment'])
            color = seg_map.get(cid, {}).get('color', '#888')
            st.markdown(
                f"<div style='padding:12px; margin-bottom:8px; border-radius:6px; "
                f"border-left:5px solid {color}; background:{color}18'>"
                f"<b style='color:{color}'>{seg}</b> &nbsp;|&nbsp; "
                f"Count: {int(row['Count']):,} &nbsp;|&nbsp; "
                f"Avg Recency: {row['Avg_Recency']:.0f}d &nbsp;|&nbsp; "
                f"Avg Frequency: {row['Avg_Frequency']:.1f} &nbsp;|&nbsp; "
                f"Avg Monetary: £{row['Avg_Monetary']:,.0f}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("""
### Technology Stack
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualisation**: matplotlib, seaborn, plotly
- **App Framework**: Streamlit
- **Model Persistence**: joblib

### Files
| File | Description |
|------|-------------|
| `Customer_Segmentation_Analysis.ipynb` | Jupyter notebook with full analysis |
| `app.py` | This Streamlit application |
| `kmeans_model.pkl` | Trained K-Means model |
| `scaler.pkl` | Fitted StandardScaler |
| `rfm_customers.csv` | Customer RFM data with segments |
| `cluster_profiles.csv` | Cluster summary statistics |
""")
