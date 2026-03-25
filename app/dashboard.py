"""
EduShield UG - School Dropout Risk Predictor & Intervention System
Main Streamlit Dashboard
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add parent directory so we can import recommender
sys.path.insert(0, os.path.dirname(__file__))
from recommender import (
    get_interventions,
    get_risk_level,
    REGION_NAMES,
    DISTRICT_SAMPLES,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EduShield Uganda",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")


@st.cache_data
def load_poverty_data():
    path = os.path.join(DATA_DIR, "UGA_2019_UNPS_v03_M_CSV", "pov2019_20.csv")
    df = pd.read_csv(path)
    # Create synthetic dropout label based on risk factors
    np.random.seed(42)
    # Higher risk: poor, rural, large household, low welfare quintile
    risk_score = (
        df["poor_2020"].astype(float) * 0.35
        + (1 - df["urban"].astype(float)) * 0.20
        + (df["hsize"].clip(upper=15) / 15) * 0.20
        + ((6 - df["quints"].clip(upper=5)) / 5) * 0.25
    )
    noise = np.random.normal(0, 0.10, len(df))
    prob = np.clip(risk_score + noise, 0, 1)
    df["dropout_risk"] = prob
    df["dropout"] = (prob >= 0.50).astype(int)
    df["region_name"] = df["region"].map(REGION_NAMES).fillna("Unknown")
    return df


@st.cache_data
def load_unesco_data():
    path = os.path.join(
        DATA_DIR, "indicator-data-export_ROFST.1T3.GPIA.CP", "data.json"
    )
    with open(path) as f:
        raw = json.load(f)
    records = raw.get("records", raw) if isinstance(raw, dict) else raw
    df = pd.DataFrame(records)
    return df


@st.cache_resource
def train_model(df):
    features = ["region", "urban", "hsize", "poor_2020", "quints", "welfare"]
    X = df[features].copy()
    # Normalize welfare to 0-1 range
    wmin, wmax = X["welfare"].min(), X["welfare"].max()
    if wmax > wmin:
        X["welfare"] = (X["welfare"] - wmin) / (wmax - wmin)
    y = df["dropout"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc, features, wmin, wmax


# ---------------------------------------------------------------------------
# Load data & model
# ---------------------------------------------------------------------------
df = load_poverty_data()
unesco_df = load_unesco_data()
model, model_accuracy, FEATURES, welfare_min, welfare_max = train_model(df)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ EduShield Uganda")
st.sidebar.markdown("*Protecting every child's right to education*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Risk Prediction", "Regional Analytics", "Early Warning"],
    index=0,
)

st.sidebar.divider()
st.sidebar.caption(f"Model accuracy: {model_accuracy:.1%}")
st.sidebar.caption(f"Households analysed: {len(df):,}")
st.sidebar.caption("Data: Uganda National Panel Survey 2019/20")

# ---------------------------------------------------------------------------
# Filters (shared)
# ---------------------------------------------------------------------------
with st.sidebar.expander("Filters", expanded=False):
    filter_region = st.multiselect(
        "Region",
        options=sorted(df["region_name"].unique()),
        default=sorted(df["region_name"].unique()),
    )
    filter_urban = st.selectbox(
        "Location", ["All", "Urban", "Rural"], index=0
    )
    filter_poverty = st.selectbox(
        "Poverty Status", ["All", "Poor", "Non-Poor"], index=0
    )

# Apply filters
filtered = df[df["region_name"].isin(filter_region)].copy()
if filter_urban == "Urban":
    filtered = filtered[filtered["urban"] == 1]
elif filter_urban == "Rural":
    filtered = filtered[filtered["urban"] == 0]
if filter_poverty == "Poor":
    filtered = filtered[filtered["poor_2020"] == 1]
elif filter_poverty == "Non-Poor":
    filtered = filtered[filtered["poor_2020"] == 0]


# =========================================================================
# PAGE: OVERVIEW
# =========================================================================
if page == "Overview":
    st.title("🛡️ EduShield Uganda Dashboard")
    st.markdown(
        "Identifying students at risk of dropping out and recommending "
        "targeted interventions using national survey data."
    )

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    total = len(filtered)
    at_risk = filtered["dropout"].sum()
    poverty_rate = filtered["poor_2020"].mean()
    avg_hsize = filtered["hsize"].mean()

    col1.metric("Total Households", f"{total:,}")
    col2.metric("At-Risk Households", f"{at_risk:,}", f"{at_risk/total:.1%}")
    col3.metric("Poverty Rate", f"{poverty_rate:.1%}")
    col4.metric("Avg Household Size", f"{avg_hsize:.1f}")

    st.divider()

    # Charts row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Dropout Risk by Region")
        region_stats = (
            filtered.groupby("region_name")
            .agg(
                total=("dropout", "count"),
                at_risk=("dropout", "sum"),
                risk_rate=("dropout", "mean"),
            )
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#ff4b4b" if r >= 0.5 else "#ffa726" if r >= 0.3 else "#66bb6a"
                  for r in region_stats["risk_rate"]]
        bars = ax.bar(region_stats["region_name"], region_stats["risk_rate"], color=colors)
        ax.set_ylabel("Dropout Risk Rate")
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, region_stats["risk_rate"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.0%}", ha="center", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

    with chart_col2:
        st.subheader("Urban vs Rural Comparison")
        loc_stats = (
            filtered.groupby("urban")
            .agg(
                total=("dropout", "count"),
                at_risk=("dropout", "sum"),
                risk_rate=("dropout", "mean"),
                poverty_rate=("poor_2020", "mean"),
            )
            .reset_index()
        )
        loc_stats["location"] = loc_stats["urban"].map({0: "Rural", 1: "Urban"})
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(loc_stats))
        w = 0.35
        ax.bar(x - w/2, loc_stats["risk_rate"], w, label="Dropout Risk", color="#ff7043")
        ax.bar(x + w/2, loc_stats["poverty_rate"], w, label="Poverty Rate", color="#42a5f5")
        ax.set_xticks(x)
        ax.set_xticklabels(loc_stats["location"])
        ax.set_ylim(0, 1)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # Wealth quintile breakdown
    st.subheader("Dropout Risk by Wealth Quintile")
    quint_stats = (
        filtered.groupby("quints")
        .agg(risk_rate=("dropout", "mean"), count=("dropout", "count"))
        .reset_index()
    )
    quint_stats["quintile"] = quint_stats["quints"].map(
        {1: "Q1 (Poorest)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (Richest)"}
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    colors_q = ["#d32f2f", "#f57c00", "#fbc02d", "#7cb342", "#2e7d32"]
    ax.bar(quint_stats["quintile"], quint_stats["risk_rate"], color=colors_q)
    ax.set_ylabel("Dropout Risk Rate")
    ax.set_ylim(0, 1)
    for i, (q, r) in enumerate(zip(quint_stats["quintile"], quint_stats["risk_rate"])):
        ax.text(i, r + 0.02, f"{r:.0%}", ha="center", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)


# =========================================================================
# PAGE: RISK PREDICTION
# =========================================================================
elif page == "Risk Prediction":
    st.title("🔮 Individual Risk Assessment")
    st.markdown("Enter student details to predict dropout risk and get intervention recommendations.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Student Details")
            age = st.slider("Age", 6, 18, 12)
            gender = st.selectbox("Gender", ["Male", "Female"])
            region = st.selectbox(
                "Region",
                options=list(REGION_NAMES.values()),
                index=0,
            )

        with col2:
            st.subheader("Household Details")
            urban = st.selectbox("Location", ["Rural", "Urban"])
            hsize = st.slider("Household Size", 1, 15, 5)
            poor = st.selectbox("Poverty Status", ["Non-Poor", "Poor"])
            welfare_quintile = st.slider("Wealth Quintile (1=Poorest, 5=Richest)", 1, 5, 3)

        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    if submitted:
        region_code = {v: k for k, v in REGION_NAMES.items()}.get(region, 1)
        urban_code = 1 if urban == "Urban" else 0
        poor_code = 1 if poor == "Poor" else 0

        # Prepare features for model
        welfare_val = welfare_min + (welfare_quintile / 5) * (welfare_max - welfare_min)
        welfare_norm = (welfare_val - welfare_min) / (welfare_max - welfare_min) if welfare_max > welfare_min else 0

        input_df = pd.DataFrame([{
            "region": region_code,
            "urban": urban_code,
            "hsize": hsize,
            "poor_2020": poor_code,
            "quints": welfare_quintile,
            "welfare": welfare_norm,
        }])

        prob = model.predict_proba(input_df)[0][1]
        risk_label, risk_color = get_risk_level(prob)

        st.divider()

        # Results
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown(f"### Risk Level: :{risk_label.lower()}[{risk_label}]" if risk_label == "Low" else f"### Risk Level")
            st.markdown(
                f'<div style="background-color:{risk_color};color:white;padding:30px;'
                f'border-radius:10px;text-align:center;font-size:24px;font-weight:bold;">'
                f'{risk_label} Risk<br><span style="font-size:48px;">{prob:.0%}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with res_col2:
            st.markdown("### Recommended Interventions")
            risk_factors = {
                "poor": poor_code,
                "urban": urban_code,
                "hsize": hsize,
                "welfare_quintile": welfare_quintile,
                "region": region_code,
                "age": age,
                "gender": gender,
            }
            interventions = get_interventions(risk_factors)

            for iv in interventions:
                priority_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(iv["priority"], "⚪")
                with st.expander(f"{priority_icon} {iv['intervention']} — *{iv['category']}*"):
                    st.write(iv["description"])
                    st.caption(f"Priority: {iv['priority']}")


# =========================================================================
# PAGE: REGIONAL ANALYTICS
# =========================================================================
elif page == "Regional Analytics":
    st.title("🗺️ Regional Analytics")

    tab1, tab2, tab3 = st.tabs(["Regional Breakdown", "Poverty Distribution", "UNESCO Trends"])

    with tab1:
        st.subheader("Dropout Risk by Region and Location")
        cross = (
            filtered.groupby(["region_name", "urban"])
            .agg(risk_rate=("dropout", "mean"), count=("dropout", "count"))
            .reset_index()
        )
        cross["location"] = cross["urban"].map({0: "Rural", 1: "Urban"})

        fig, ax = plt.subplots(figsize=(10, 5))
        regions = sorted(cross["region_name"].unique())
        x = np.arange(len(regions))
        w = 0.35
        for i, loc in enumerate(["Rural", "Urban"]):
            subset = cross[cross["location"] == loc].set_index("region_name").reindex(regions)
            vals = subset["risk_rate"].fillna(0).values
            color = "#e57373" if loc == "Rural" else "#64b5f6"
            ax.bar(x + (i - 0.5) * w, vals, w, label=loc, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(regions)
        ax.set_ylabel("Dropout Risk Rate")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

        # Summary table
        summary = (
            filtered.groupby("region_name")
            .agg(
                households=("dropout", "count"),
                at_risk=("dropout", "sum"),
                risk_rate=("dropout", "mean"),
                poverty_rate=("poor_2020", "mean"),
                avg_hh_size=("hsize", "mean"),
            )
            .reset_index()
        )
        summary["risk_rate"] = summary["risk_rate"].map("{:.1%}".format)
        summary["poverty_rate"] = summary["poverty_rate"].map("{:.1%}".format)
        summary["avg_hh_size"] = summary["avg_hh_size"].map("{:.1f}".format)
        summary.columns = ["Region", "Households", "At Risk", "Risk Rate", "Poverty Rate", "Avg HH Size"]
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Poverty Distribution Across Regions")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Poverty rate by region
        pov_by_region = filtered.groupby("region_name")["poor_2020"].mean().sort_values(ascending=False)
        axes[0].barh(pov_by_region.index, pov_by_region.values, color="#ef5350")
        axes[0].set_xlabel("Poverty Rate")
        axes[0].set_title("Poverty Rate by Region")
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # Wealth quintile distribution
        quint_counts = filtered["quints"].value_counts().sort_index()
        labels = ["Q1\n(Poorest)", "Q2", "Q3", "Q4", "Q5\n(Richest)"]
        colors_q = ["#d32f2f", "#f57c00", "#fbc02d", "#7cb342", "#2e7d32"]
        axes[1].pie(quint_counts.values, labels=labels, colors=colors_q,
                    autopct="%1.0f%%", startangle=90)
        axes[1].set_title("Wealth Quintile Distribution")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.subheader("UNESCO Out-of-School Rate Trends")
        # Filter for Uganda and neighboring countries
        countries = ["UGA", "KEN", "TZA", "RWA", "COD", "SSD"]
        country_names = {
            "UGA": "Uganda", "KEN": "Kenya", "TZA": "Tanzania",
            "RWA": "Rwanda", "COD": "DR Congo", "SSD": "South Sudan"
        }
        unesco_filtered = unesco_df[unesco_df["geoUnit"].isin(countries)].copy()
        unesco_filtered["country"] = unesco_filtered["geoUnit"].map(country_names)

        if not unesco_filtered.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            for country in unesco_filtered["country"].unique():
                subset = unesco_filtered[unesco_filtered["country"] == country].sort_values("year")
                style = "-" if country == "Uganda" else "--"
                lw = 3 if country == "Uganda" else 1.5
                ax.plot(subset["year"], subset["value"], style, linewidth=lw, label=country)
            ax.set_xlabel("Year")
            ax.set_ylabel("Gender Parity Index (Out-of-School)")
            ax.set_title("Out-of-School Gender Parity Index — East Africa")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No UNESCO data available for selected countries.")


# =========================================================================
# PAGE: EARLY WARNING
# =========================================================================
elif page == "Early Warning":
    st.title("⚠️ Early Warning System")
    st.markdown("Households with the highest predicted dropout risk requiring immediate attention.")

    # High-risk households
    high_risk = filtered[filtered["dropout_risk"] >= 0.7].copy()
    medium_risk = filtered[(filtered["dropout_risk"] >= 0.4) & (filtered["dropout_risk"] < 0.7)].copy()

    # Alert summary
    alert_col1, alert_col2, alert_col3 = st.columns(3)
    alert_col1.metric("🔴 High Risk", f"{len(high_risk):,}")
    alert_col2.metric("🟡 Medium Risk", f"{len(medium_risk):,}")
    alert_col3.metric("🟢 Low Risk", f"{len(filtered) - len(high_risk) - len(medium_risk):,}")

    st.divider()

    # Top at-risk households table
    st.subheader("Top 50 Highest-Risk Households")
    top_risk = (
        filtered.nlargest(50, "dropout_risk")[
            ["hhid", "region_name", "urban", "hsize", "poor_2020", "quints", "dropout_risk"]
        ]
        .copy()
    )
    top_risk["urban"] = top_risk["urban"].map({0: "Rural", 1: "Urban"})
    top_risk["poor_2020"] = top_risk["poor_2020"].map({0: "No", 1: "Yes"})
    top_risk["dropout_risk"] = top_risk["dropout_risk"].map("{:.0%}".format)
    top_risk["quints"] = top_risk["quints"].map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5"})
    top_risk.columns = ["Household ID", "Region", "Location", "HH Size", "Poor", "Quintile", "Risk Score"]
    # Truncate household IDs for display
    top_risk["Household ID"] = top_risk["Household ID"].str[:12] + "..."
    st.dataframe(top_risk, use_container_width=True, hide_index=True)

    st.divider()

    # Risk distribution
    st.subheader("Risk Score Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(filtered["dropout_risk"], bins=30, color="#42a5f5", edgecolor="white")
    ax.axvline(0.7, color="#ff4b4b", linestyle="--", linewidth=2, label="High Risk Threshold (70%)")
    ax.axvline(0.4, color="#ffa726", linestyle="--", linewidth=2, label="Medium Risk Threshold (40%)")
    ax.set_xlabel("Dropout Risk Score")
    ax.set_ylabel("Number of Households")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

    # Intervention needs summary
    st.subheader("Intervention Needs Summary")
    needs = {"School Feeding": 0, "Cash Transfer": 0, "Transport Support": 0,
             "Scholarships": 0, "Free Materials": 0}
    needs["School Feeding"] = int(high_risk["poor_2020"].eq("Yes").sum() if "poor_2020" in high_risk.columns else 0)
    needs["Cash Transfer"] = needs["School Feeding"]
    needs["Transport Support"] = int((filtered["urban"] == 0).sum() * filtered["dropout"].mean())
    needs["Scholarships"] = int((filtered["hsize"] > 6).sum() * filtered["dropout"].mean())
    needs["Free Materials"] = int((filtered["quints"] <= 2).sum() * filtered["dropout"].mean())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(list(needs.keys()), list(needs.values()), color=["#ef5350", "#ff7043", "#ffa726", "#66bb6a", "#42a5f5"])
    ax.set_xlabel("Estimated Households Needing Intervention")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.markdown(
    "Built by **Raymond Wayesu**  \n"
    "Biostatistician & Data Scientist  \n"
    "Data: UNPS 2019/20 | UNESCO UIS"
)
