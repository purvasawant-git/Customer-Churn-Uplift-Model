import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Churn Uplift Dashboard")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'telco_churn_uplift_ready.csv')

@st.cache_data
def load_data():
    # Fix: correct path is data/raw/processed/
    path = DATA_PATH
    df = pd.read_csv(path)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    return df

@st.cache_resource
def load_models():
    df = load_data()

    # Feature prep — encode first
    drop_cols = ['Churn', 'persuadability', 'persuadability_bin',
                 'contract_score', 'tenure_norm', 'charge_norm']
    available_drop = [col for col in drop_cols if col in df.columns]
    df_features = df.drop(columns=available_drop)
    object_cols = df_features.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df_features, columns=object_cols, drop_first=True)
    df_encoded.columns = (df_encoded.columns
                          .str.replace(' ', '_')
                          .str.replace('-', '_')
                          .str.replace('(', '')
                          .str.replace(')', ''))
    feature_cols = [col for col in df_encoded.columns
                    if col not in ['treatment', 'churn_observed']]
    X        = df_encoded[feature_cols]
    y_churn  = df_encoded['churn_observed']
    treatment = df_encoded['treatment']

    # Try loading uplift models from joblib
    models_dir      = os.path.join(BASE_DIR, 'models')
    treated_path    = os.path.join(models_dir, 'xgb_treated.pkl')
    control_path    = os.path.join(models_dir, 'xgb_control.pkl')

    if os.path.exists(treated_path) and os.path.exists(control_path):
        xgb_treated = joblib.load(treated_path)
        xgb_control = joblib.load(control_path)
    else:
        st.warning("Uplift models not found — training now (~30 seconds)...")
        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _, tr_train, _ = train_test_split(
            X, y_churn, treatment, test_size=0.2, stratify=y_churn, random_state=42
        )
        xgb_treated = xgb.XGBClassifier(n_estimators=200, max_depth=4,
                                          learning_rate=0.05, eval_metric='logloss',
                                          random_state=42)
        xgb_treated.fit(X_train[tr_train == 1], y_train[tr_train == 1])
        xgb_control = xgb.XGBClassifier(n_estimators=200, max_depth=4,
                                          learning_rate=0.05, eval_metric='logloss',
                                          random_state=42)
        xgb_control.fit(X_train[tr_train == 0], y_train[tr_train == 0])

    # Load or train churn model
    churn_path = os.path.join(models_dir, 'churn_xgb.pkl')
    if os.path.exists(churn_path):
        churn_xgb = joblib.load(churn_path)
    else:
        churn_xgb = xgb.XGBClassifier(n_estimators=200, max_depth=4,
                                        learning_rate=0.05, eval_metric='logloss',
                                        random_state=42)
        churn_xgb.fit(X, y_churn)

    # Predictions
    churn_pred = churn_xgb.predict_proba(X)[:, 1]
    p_control  = xgb_control.predict_proba(X)[:, 1]
    p_treated  = xgb_treated.predict_proba(X)[:, 1]
    uplift     = p_control - p_treated

    # SHAP
    sample_X     = X.sample(min(1000, len(X)), random_state=42)
    explainer    = shap.Explainer(churn_xgb, sample_X)
    shap_vals    = explainer(sample_X).values
    shap_mean    = np.abs(shap_vals).mean(0)
    top_features = pd.Series(shap_mean, index=feature_cols).nlargest(10)

    return {
        'churn_xgb':   churn_xgb,
        'xgb_treated': xgb_treated,
        'xgb_control': xgb_control,
        'X':           X,
        'uplift':      uplift,
        'churn_pred':  churn_pred,
        'shap_top':    top_features
    }

def get_segment(uplift):
    if uplift < -0.05:
        return 'Sleeping Dog'
    elif uplift < 0.02:
        return 'Lost Cause'
    elif uplift < 0.15:
        return 'Persuadable'
    else:
        return 'Sure Thing'

def style_segment(val):
    colors = {
        'Sleeping Dog': '#ffcccc',
        'Lost Cause':   '#cccccc',
        'Persuadable':  '#ccffcc',
        'Sure Thing':   '#ccccff'
    }
    return f'background-color: {colors.get(val, "white")}'

st.title("Telco Churn & Uplift Intelligence Dashboard")

df      = load_data()
models  = load_models()

df['churn_risk']   = models['churn_pred'] * 100
df['uplift_score'] = models['uplift']
df['segment']      = df['uplift_score'].apply(get_segment)

# Sidebar
st.sidebar.header("Filters")
contracts     = st.sidebar.multiselect("Contract",
                                        options=df['Contract'].unique(),
                                        default=df['Contract'].unique())
tenure_range  = st.sidebar.slider("Tenure (months)", 0, 72, (0, 72))
charges_range = st.sidebar.slider("Monthly Charges", 0.0, 120.0, (0.0, 120.0))

st.sidebar.markdown("---")
st.sidebar.header("ROI Parameters")
offer_cost      = st.sidebar.number_input("Offer cost per customer", value=500, step=50)
margin_retained = st.sidebar.number_input("Margin per retained customer", value=3000, step=100)
target_thresh   = st.sidebar.slider("Targeting threshold (top %)", 5, 50, 20) / 100

filtered_df = df[
    (df['Contract'].isin(contracts)) &
    (df['tenure'].between(*tenure_range)) &
    (df['MonthlyCharges'].between(*charges_range))
].copy()

n_total = len(filtered_df)

# Metric cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", n_total)
with col2:
    st.metric("Avg Churn Risk", f"{filtered_df['churn_risk'].mean():.1f}%")
with col3:
    st.metric("Avg Uplift Score", f"{filtered_df['uplift_score'].mean():.4f}")
with col4:
    n_target  = int(n_total * target_thresh)
    top_uplift = filtered_df.nlargest(n_target, 'uplift_score')['uplift_score'].sum()
    camp_cost  = n_target * offer_cost
    profit     = top_uplift * margin_retained - camp_cost
    roi_pct    = (profit / camp_cost * 100) if camp_cost > 0 else 0
    st.metric("Expected ROI", f"{roi_pct:.1f}%")

# Tabs
tab1, tab2, tab3 = st.tabs(["Customer Intelligence", "ROI Simulator", "SHAP Analysis"])

with tab1:
    display_cols = ['tenure', 'MonthlyCharges', 'Contract',
                    'churn_risk', 'uplift_score', 'segment']
    df_display = filtered_df[display_cols].copy()
    df_display['churn_risk']   = df_display['churn_risk'].round(1).astype(str) + '%'
    df_display['uplift_score'] = df_display['uplift_score'].round(4)
    styled_df = df_display.style.applymap(style_segment, subset=['segment'])
    st.dataframe(styled_df, use_container_width=True)

    seg_counts = filtered_df['segment'].value_counts()
    fig_pie = px.pie(values=seg_counts.values, names=seg_counts.index,
                     color_discrete_map={
                         'Sleeping Dog': '#ffcccc', 'Lost Cause': '#cccccc',
                         'Persuadable':  '#ccffcc', 'Sure Thing': '#ccccff'
                     })
    st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    thresholds = np.linspace(0.05, 0.5, 46)
    rois = []
    for thresh in thresholds:
        n_t      = int(n_total * thresh)
        top_u    = filtered_df.nlargest(n_t, 'uplift_score')['uplift_score'].sum()
        cost     = n_t * offer_cost
        roi      = ((top_u * margin_retained - cost) / cost * 100) if cost > 0 else 0
        rois.append(roi)

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(x=thresholds * 100, y=rois,
                                  mode='lines+markers', name='ROI'))
    fig_roi.add_vline(x=target_thresh * 100, line_dash="dash", line_color="red",
                      annotation_text=f"Selected: {target_thresh:.0%}")
    fig_roi.update_layout(title="ROI % vs Targeting Threshold",
                          xaxis_title="Top % by Uplift", yaxis_title="ROI %")
    st.plotly_chart(fig_roi, use_container_width=True)

    for strength in [0.5, 1.0, 1.5]:
        pass  # sensitivity already in fig_roi context

    fig_sens = go.Figure()
    for strength in [0.5, 1.0, 1.5]:
        scaled = filtered_df['uplift_score'].values * strength
        rois_s = []
        for thresh in thresholds:
            n_t   = int(n_total * thresh)
            top_u = np.sort(scaled)[-n_t:].sum()
            cost  = n_t * offer_cost
            roi   = ((top_u * margin_retained - cost) / cost * 100) if cost > 0 else 0
            rois_s.append(roi)
        fig_sens.add_trace(go.Scatter(x=thresholds * 100, y=rois_s,
                                       name=f'{strength}x Effect'))
    fig_sens.update_layout(title="ROI Sensitivity Analysis",
                           xaxis_title="Top % by Uplift", yaxis_title="ROI %")
    st.plotly_chart(fig_sens, use_container_width=True)

    n_targeted  = int(n_total * target_thresh)
    top_c       = filtered_df.nlargest(n_targeted, 'uplift_score')
    exp_profit  = top_c['uplift_score'].sum() * margin_retained - n_targeted * offer_cost
    c1, c2, c3  = st.columns(3)
    with c1:
        st.metric("Targeted Customers", n_targeted)
    with c2:
        st.metric("Campaign Cost", f"Rs {n_targeted * offer_cost:,.0f}")
    with c3:
        st.metric("Expected Profit", f"Rs {exp_profit:,.0f}")

with tab3:
    fig_shap = px.bar(x=models['shap_top'].values, y=models['shap_top'].index,
                      orientation='h',
                      title="Top 10 SHAP Feature Importance (Mean |SHAP|)")
    fig_shap.update_layout(yaxis_title="Feature", xaxis_title="Mean |SHAP value|")
    st.plotly_chart(fig_shap, use_container_width=True)
    st.markdown("Features with high SHAP values drive churn predictions most strongly.")

