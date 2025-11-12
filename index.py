# index.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

# === STREAMLIT CONFIG ===
st.set_page_config(page_title="Ocean Health Index Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

# === SIDEBAR ===
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your OHI CSV file", type=["csv"])

# === LOAD & CLEAN DATA ===
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df.drop(columns=['Shape__Area', 'Shape__Length'], errors='ignore')
    df = df.rename(columns={
        'rgn_nam': 'region_name',
        'Index_': 'OHI_index',
        'trnd_sc': 'trend_score'
    })
    return df

if uploaded_file is None:
    st.warning("Please upload your dataset (e.g. OHI_final_formatted_scores_2020-10-01.csv).")
    st.stop()

df = load_data(uploaded_file)

# === PAGE TITLE ===
st.title("Ocean Health Index Dashboard")
st.markdown("Gain insights into ocean sustainability indicators and explore correlations between ecological, social, and economic goals.")

# === TABS ===
tab_overview, tab_corr, tab_ml = st.tabs(["Overview", "Correlations", "Machine Learning"])

# -----------------
# OVERVIEW TAB
# -----------------
with tab_overview:
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.dataframe(df.head(), use_container_width=True)

    # Distribution plot
    st.subheader("Distribution of Ocean Health Index")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["OHI_index"], bins=20, kde=True, color="teal", ax=ax)
    ax.set_xlabel("OHI Index")
    ax.set_ylabel("Number of Regions")
    st.pyplot(fig)

    # Top 10
    st.subheader("Top 10 Regions by OHI Index")
    top10 = df.nlargest(10, "OHI_index")[["region_name", "OHI_index"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top10, x="OHI_index", y="region_name", palette="viridis", ax=ax)
    ax.set_title("Top 10 Ocean Health Regions")
    st.pyplot(fig)

# --------------------
# CORRELATION TAB
# --------------------
with tab_corr:
    st.subheader("Goal Correlation Analysis")
    goal_columns = ['AO', 'BD', 'CP', 'CS', 'CW', 'ECO', 'FIS', 'FP',
                    'HAB', 'ICO', 'LE', 'LIV', 'LSP', 'MAR', 'NP', 'SP',
                    'SPP', 'TR']

    # Overall heatmap
    st.write("### Overall Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df[goal_columns + ["OHI_index"]].corr(),
                cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

    with st.expander("Focused Correlation Heatmap"):
        selected_goal = st.selectbox("Select a goal to inspect:", goal_columns)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            df[goal_columns + ["OHI_index", "trend_score"]]
            .corr()[[selected_goal]]
            .sort_values(by=selected_goal, ascending=False),
            annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax
        )
        ax.set_title(f"Correlations of {selected_goal} with Other Goals")
        st.pyplot(fig)

# ---------------------
# MACHINE LEARNING TAB
# ---------------------
with tab_ml:
    st.subheader("Machine Learning: Predicting OHI Index")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    excluded = ['OBJECTID', 'rgn_id', 'are_km2', 'OHI_index', 'trend_score']
    goal_columns = [col for col in numeric_cols if col not in excluded]

    if len(goal_columns) < 5:
        st.error("Not enough numeric goal columns found for training the model.")
        st.stop()

    X = df[goal_columns]
    y = df["OHI_index"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    # --- TRAIN BUTTON ---
    if st.button("Train AI Model"):
        model = RandomForestRegressor(n_estimators=100000, random_state=2000)
        model.fit(X_train, y_train)

        # Save model in session
        st.session_state["model"] = model
        st.session_state["goal_columns"] = goal_columns

        # Predictions
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.success(f"Model trained successfully! R² = {r2:.3f}, MAE = {mae:.2f}")

        # === Predicted vs Actual ===
        st.subheader("Predicted vs Actual OHI Index")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "--", linewidth=1, color="gray")
        ax.set_xlabel("Actual OHI Index")
        ax.set_ylabel("Predicted OHI Index")
        st.pyplot(fig)

        # === Residuals histogram ===
        st.subheader("Residuals (Prediction Error) Distribution")
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, bins=20, kde=True, ax=ax, color="orange")
        ax.set_xlabel("Residual (Actual - Predicted)")
        st.pyplot(fig)

        # === Feature importance ===
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            "Goal": goal_columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.dataframe(importance, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=importance, x="Importance", y="Goal", palette="viridis", ax=ax)
        ax.set_title("Contribution of Each Goal to OHI Prediction")
        st.pyplot(fig)

    # --- PREDICT NEW DATA ---
    st.subheader("Predict Future OHI Index (Using Trained Model)")
    uploaded_new_data = st.file_uploader(
        "Upload new regional data (CSV) for prediction", type=["csv"], key="predict_upload"
    )

    if uploaded_new_data is not None:
        new_df = pd.read_csv(uploaded_new_data)
        st.write("New data loaded:")
        st.dataframe(new_df.head(), use_container_width=True)

        if "model" not in st.session_state:
            st.warning("Train the model first before predicting.")
        else:
            model = st.session_state["model"]
            goal_columns = st.session_state["goal_columns"]

            missing = [f for f in goal_columns if f not in new_df.columns]
            if missing:
                st.error(f"Missing features in new dataset: {missing}")
            else:
                X_new = new_df[goal_columns]
                preds = model.predict(X_new)
                new_df["Predicted_OHI_Index"] = preds

                st.success("Predictions complete!")
                st.dataframe(new_df, use_container_width=True)

                # Distributie
                st.subheader("Distribution of Predicted OHI Index")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(new_df["Predicted_OHI_Index"], bins=20, kde=True, color="teal", ax=ax)
                st.pyplot(fig)

                # Optional region plot
                if "region_name" in new_df.columns:
                    st.subheader("Predicted OHI Index by Region")
                    top_plot = new_df.sort_values("Predicted_OHI_Index", ascending=False).head(15)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(data=top_plot, x="Predicted_OHI_Index", y="region_name", palette="viridis", ax=ax)
                    st.pyplot(fig)
