# ui_pollution.py
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="🌊 Ocean Health Index Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

# --- SIDEBAR: FILE UPLOAD ---
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your OHI CSV file", type=["csv"])

# --- LOAD & CLEAN DATA ---
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
    st.warning("👈 Upload your OHI dataset (e.g. OHI_final_formatted_scores_2020-10-01.csv)")
    st.stop()

df = load_data(uploaded_file)
st.success("✅ Dataset loaded successfully!")

# --- OVERVIEW ---
st.header("📊 Dataset Overview")
st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
st.dataframe(df.head())

st.subheader("Missing values per column")
st.write(df.isnull().sum())

# --- DISTRIBUTION OF OHI INDEX ---
st.header("🌍 Distribution of Ocean Health Index")
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["OHI_index"], bins=20, kde=True, color="skyblue", ax=ax)
ax.set_xlabel("OHI Index")
ax.set_ylabel("Number of Regions")
ax.set_title("Distribution of Ocean Health Index Scores")
st.pyplot(fig)

# --- TOP 10 REGIONS ---
st.header("🏆 Top 10 Regions by OHI Index")
top10 = df.nlargest(10, "OHI_index")[["region_name", "OHI_index"]]
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=top10, x="OHI_index", y="region_name", palette="viridis", ax=ax)
ax.set_title("Top 10 Regions by Ocean Health Index")
st.pyplot(fig)

# --- CORRELATION HEATMAP ---
st.header("🔗 Correlation Between OHI Goal Scores")
goal_columns = ['AO', 'BD', 'CP', 'CS', 'CW', 'ECO', 'FIS', 'FP', 'HAB', 'ICO',
                'LE', 'LIV', 'LSP', 'MAR', 'NP', 'SP', 'SPP', 'TR']
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df[goal_columns].corr(), cmap="coolwarm", ax=ax)
ax.set_title("Correlation Between OHI Goal Scores")
st.pyplot(fig)

# --- SCATTER: OHI vs TREND ---
st.header("📈 OHI Index vs Trend Score")
fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(data=df, x="OHI_index", y="trend_score", ax=ax)
ax.set_title("OHI Index vs Trend Score")
st.pyplot(fig)

# --- SUBGOAL BREAKDOWN ---
st.header("🌐 Sub-goal Breakdown for a Specific Region")
region = st.selectbox("Select a region:", sorted(df["region_name"].unique()))
subset = df[df["region_name"] == region]
if not subset.empty:
    scores = subset.iloc[0][goal_columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    scores.plot(kind="bar", color="teal", ax=ax)
    ax.set_title(f"{region} — Sub-goal Scores")
    ax.set_ylabel("Score")
    st.pyplot(fig)

# === 🧠 MACHINE LEARNING & AI PREDICTION SECTION ===
st.header("🤖 Machine Learning: Predicting OHI Index")

# Detect numeric features (goal columns)
numeric_cols = df.select_dtypes(include='number').columns.tolist()
excluded_cols = ['OBJECTID', 'rgn_id', 'are_km2', 'OHI_index', 'trend_score']
goal_columns = [col for col in numeric_cols if col not in excluded_cols]

if len(goal_columns) < 5:
    st.error("❌ Not enough numeric goal columns found for training the model.")
    st.write("Detected numeric columns:", numeric_cols)
    st.stop()

# Train/test split
X = df[goal_columns]
y = df["OHI_index"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- TRAIN MODEL BUTTON ---
if st.button("🚀 Train AI Model"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Display results
    st.success("✅ Model trained successfully!")
    st.write(f"**R² score:** {r2:.3f}")
    st.write(f"**Mean Absolute Error:** {mae:.2f}")

    # Save model
    joblib.dump(model, "trained_model.pkl")
    joblib.dump(goal_columns, "model_features.pkl")
    st.info("💾 Model and feature names saved (trained_model.pkl, model_features.pkl)")

    # Show importance (coefficients)
    importance = pd.DataFrame({
        "Goal": goal_columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)

    st.subheader("📈 Top contributing goals to OHI Index")
    st.dataframe(importance)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=importance, x="Coefficient", y="Goal", palette="viridis", ax=ax)
    ax.set_title("Contribution of Each Goal to Ocean Health Index")
    st.pyplot(fig)


# --- PREDICTION SECTION ---
st.header("🔮 Predict Future OHI Index (Using Trained Model)")

if not os.path.exists("trained_model.pkl"):
    st.warning("⚠️ No trained model found. Please train the AI model first.")
else:
    uploaded_new_data = st.file_uploader("Upload new regional data (CSV) for prediction", type=["csv"], key="predict_upload")

    if uploaded_new_data is not None:
        new_df = pd.read_csv(uploaded_new_data)
        st.write("📄 New data loaded:")
        st.dataframe(new_df.head())

        # Load model and features
        model = joblib.load("trained_model.pkl")
        feature_names = joblib.load("model_features.pkl")

        # Check if all required features exist
        missing_features = [f for f in feature_names if f not in new_df.columns]
        if missing_features:
            st.error(f"❌ Missing features in new dataset: {missing_features}")
        else:
            X_new = new_df[feature_names]
            predictions = model.predict(X_new)
            new_df["Predicted_OHI_Index"] = predictions

            st.success("✅ Predictions complete!")
            st.subheader("🔮 Predicted OHI Index for New Data")
            st.dataframe(new_df)

            # Optional: show distribution of predictions
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(new_df["Predicted_OHI_Index"], bins=20, kde=True, color="teal", ax=ax)
            ax.set_title("Distribution of Predicted OHI Index")
            st.pyplot(fig)
