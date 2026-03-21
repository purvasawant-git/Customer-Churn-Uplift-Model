import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Churn Uplift Dashboard")
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'telco_churn_uplift_ready.csv'))
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    return df

@st.cache_resource
def train_models(df):
    # Feature prep matching notebook 03
    drop_cols = ['Churn', 'persuadability', 'persuadability_bin', 'contract_score', 'tenure_norm', 'charge_norm']
    available_drop = [col for col in drop_cols if col in df.columns]
    df_features = df.drop(columns=available_drop)
    
    object_cols = df_features.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df_features, columns=object_cols, drop_first=True)
    df_encoded.columns = df_encoded.columns.str.replace(' ', '_').str.replace('-', '_').str.replace('(', '').str.replace(')', '')
    
    feature_cols = [col for col in df_encoded.columns if col not in ['treatment', 'churn_observed']]
    X = df_encoded[feature_cols]
    y_churn = df_encoded['churn_observed']
    
    # Train/test split for T-Learner (80/20 as per "train on 80%")
    X_train, X_test, y_train, y_test, tr_train, tr_test = train_test_split(
        X, y_churn, df_encoded['treatment'], test_size=0.2, stratify=y_churn, random_state=42
    )
    
    # Churn XGBClassifier on full dataset
    churn_xgb = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, 
                                  eval_metric='logloss', random_state=42)
    churn_xgb.fit(X, y_churn)
    
    # T-Learner
    treated_idx = tr_train == 1
    xgb_treated = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, 
                                    eval_metric='logloss', random_state=42)
    xgb_treated.fit(X_train[treated_idx], y_train[treated_idx])
    
    control_idx = tr_train == 0
    xgb_control = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, 
                                    eval_metric='logloss', random_state=42)
    xgb_control.fit(X_train[control_idx], y_train[control_idx])
    
    # Uplift on FULL dataset
    p_treated_full = xgb_treated.predict_proba(X)[:, 1]
    p_control_full = xgb_control.predict_proba(X)[:, 1]
    uplift_full = p_control_full - p_treated_full
    
    # SHAP for churn_xgb on sample for speed (full if small)
    explainer = shap.TreeExplainer(churn_xgb)
    shap_values = explainer.shap_values(X.sample(min(1000, len(X))))
    shap_mean = np.abs(shap_values).mean(0)
    top_features = pd.Series(shap_mean, index=feature_cols).nlargest(10)
    
    return {
        'churn_xgb': churn_xgb,
        'xgb_treated': xgb_treated,
        'xgb_control': xgb_control,
        'X': X,
        'df_encoded': df_encoded,
        'df': df,
        'uplift': uplift_full,
        'churn_pred': churn_xgb.predict_proba(X)[:, 1],
        'shap_top': top_features
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
    color = {'Sleeping Dog': 'background-color: #ffcccc', 
             'Lost Cause': 'background-color: #cccccc', 
             'Persuadable': 'background-color: #ccffcc', 
             'Sure Thing': 'background-color: #ccccff'}
    return f'background-color: {color.get(val, "white")}'

st.title("🛡️ Telco Churn & Uplift Intelligence Dashboard")

df = load_data()
models = train_models(df)

df['churn_risk'] = models['churn_pred'] * 100
df['uplift_score'] = models['uplift']
df['segment'] = df['uplift_score'].apply(get_segment)

# Sidebar
st.sidebar.header("🔍 Filters")
contracts = st.sidebar.multiselect("Contract", 
                                   options=df['Contract'].unique(), 
                                   default=df['Contract'].unique())
tenure_range = st.sidebar.slider("Tenure (months)", 0, 72, (0, 72))
charges_range = st.sidebar.slider("Monthly Charges ($)", 0.0, 120.0, (0.0, 120.0))

st.sidebar.markdown("---")
st.sidebar.header("💰 ROI Parameters")
offer_cost = st.sidebar.number_input("Offer cost per customer", value=500, step=50)
margin_retained = st.sidebar.number_input("Margin per retained customer", value=3000, step=100)
target_thresh = st.sidebar.slider("Targeting threshold (top %)", 5, 50, 20) / 100

filtered_df = df[
    (df['Contract'].isin(contracts)) &
    (df['tenure'].between(*tenure_range)) &
    (df['MonthlyCharges'].between(*charges_range))
].copy()

n_total = len(filtered_df)

# Top metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", n_total)
with col2:
    st.metric("Avg Churn Risk", f"{filtered_df['churn_risk'].mean():.1f}%")
with col3:
    st.metric("Avg Uplift Score", f"{filtered_df['uplift_score'].mean():.4f}")
with col4:
    n_target = int(n_total * target_thresh)
    top_uplift = filtered_df.nlargest(n_target, 'uplift_score')['uplift_score'].sum()
    camp_cost = n_target * offer_cost
    profit = top_uplift * margin_retained - camp_cost
    roi_pct = (profit / camp_cost * 100) if camp_cost > 0 else 0
    st.metric("Expected ROI", f"{roi_pct:.1f}%")

# Tabs
tab1, tab2, tab3 = st.tabs(["👥 Customer Intelligence", "💹 ROI Simulator", "🔍 SHAP Analysis"])

with tab1:
    display_cols = ['tenure', 'MonthlyCharges', 'Contract', 'churn_risk', 'uplift_score', 'segment']
    df_display = filtered_df[display_cols].copy()
    df_display['churn_risk'] = df_display['churn_risk'].round(1).astype(str) + '%'
    df_display['uplift_score'] = df_display['uplift_score'].round(4)
    styled_df = df_display.style.format({'churn_risk': str, 'uplift_score': '{:.4f}'})
    styled_df = styled_df.applymap(style_segment, subset=['segment'])
    st.dataframe(styled_df, use_container_width=True)
    
    seg_counts = filtered_df['segment'].value_counts()
    fig_pie = px.pie(values=seg_counts.values, names=seg_counts.index, 
                     color_discrete_map={
                         'Sleeping Dog': '#ffcccc', 'Lost Cause': '#cccccc', 
                         'Persuadable': '#ccffcc', 'Sure Thing': '#ccccff'
                     })
    st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    thresholds = np.linspace(0.05, 0.5, 46)
    rois = []
    for thresh in thresholds:
        n_target_roi = int(n_total * thresh)
        top_uplift_roi = filtered_df.nlargest(n_target_roi, 'uplift_score')['uplift_score'].sum()
        cost = n_target_roi * offer_cost
        profit_roi = top_uplift_roi * margin_retained - cost
        roi = (profit_roi / cost * 100) if cost > 0 else 0
        rois.append(roi)
    
    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(x=thresholds*100, y=rois, mode='lines+markers', name='ROI'))
    fig_roi.add_vline(x=target_thresh*100, line_dash="dash", line_color="red", 
                      annotation_text=f"Selected: {target_thresh:.0%}")
    fig_roi.update_layout(title="ROI % vs Targeting Threshold", xaxis_title="Top % by Uplift", 
                          yaxis_title="ROI %", showlegend=True)
    st.plotly_chart(fig_roi, use_container_width=True)
    
    # Sensitivity
    strengths = [0.5, 1.0, 1.5]
    fig_sens = make_subplots(specs=[[{"secondary_y": False}]])
    for strength in strengths:
        scaled_uplift_sens = filtered_df['uplift_score'].values * strength
        rois_sens = []
        for thresh in thresholds:
            n_target_sens = int(n_total * thresh)
            top_uplift_sens = np.sort(scaled_uplift_sens)[-n_target_sens:].sum()
            cost_sens = n_target_sens * offer_cost
            roi_sens = ((top_uplift_sens * margin_retained - cost_sens) / cost_sens * 100) if cost_sens > 0 else 0
            rois_sens.append(roi_sens)
        fig_sens.add_trace(go.Scatter(x=thresholds*100, y=rois_sens, name=f'{strength}x Effect'))
    fig_sens.update_layout(title="ROI Sensitivity (Treatment Effect Strength)", 
                           xaxis_title="Top % by Uplift", yaxis_title="ROI %")
    st.plotly_chart(fig_sens, use_container_width=True)
    
    # Key metrics
    n_targeted = int(n_total * target_thresh)
    top_customers = filtered_df.nlargest(n_targeted, 'uplift_score')
    exp_profit = top_customers['uplift_score'].sum() * margin_retained - n_targeted * offer_cost
    col1m, col2m, col3m = st.columns(3)
    with col1m:
        st.metric("Targeted Customers", n_targeted)
    with col2m:
        st.metric("Campaign Cost", f"${n_targeted * offer_cost:,.0f}")
    with col3m:
        st.metric("Expected Profit", f"${exp_profit:,.0f}")

with tab3:
    fig_shap = px.bar(x=models['shap_top'].values, y=models['shap_top'].index, 
                      orientation='h', title="Top 10 SHAP Feature Importance (Mean |SHAP|)")
    fig_shap.update_layout(yaxis_title="Feature", xaxis_title="Mean |SHAP value|")
    st.plotly_chart(fig_shap, use_container_width=True)
    
    st.markdown("""
    **SHAP Explanation:** Features with high SHAP values drive churn predictions most strongly. 
    Positive SHAP increases churn probability, negative decreases it.
    """)

