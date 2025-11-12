import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="🌊 Ocean Health Index Dashboard", layout="wide")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# --- SIDEBAR: FILE UPLOAD ---
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your OHI CSV file", type=["csv"])

# --- LOAD DATA FUNCTION ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect if it's the "long format" (goal, dimension, value)
    if {"goal", "dimension", "value"}.issubset(df.columns):
        st.info("Detected long-format dataset. Pivoting to wide format...")
        df = df.pivot_table(
            index=["region_id", "region_name"],
            columns=["goal", "dimension"],
            values="value"
        ).reset_index()

        # Flatten MultiIndex
        df.columns = ['_'.join([str(c) for c in col if c]) for col in df.columns]
        # Rename "Index_status" to "OHI_index"
        rename_map = {col: col.replace("Index_status", "OHI_index") for col in df.columns}
        df = df.rename(columns=rename_map)
    else:
        # Normal wide-format cleaning
        rename_map = {
            'rgn_nam': 'region_name',
            'index_': 'OHI_index',
            'index': 'OHI_index',
            'ohi_index': 'OHI_index',
            'trnd_sc': 'trend_score',
            'trend': 'trend_score'
        }
        df = df.rename(columns=rename_map)
        df = df.drop(columns=['shape__area', 'shape__length'], errors='ignore')

    return df

# --- LOAD DATA ---
if uploaded_file is None:
    st.warning("👈 Upload your OHI dataset to begin.")
    st.stop()

df = load_data(uploaded_file)
st.success("✅ Dataset loaded successfully!")

# --- CHECK COLUMNS ---
st.write("📋 **Detected columns:**", list(df.columns))

if "OHI_index" not in df.columns:
    st.error("❌ Could not find the 'OHI_index' column. Please ensure your dataset includes it (or the goal 'Index' if using long format).")
    st.stop()

# --- OVERVIEW ---
st.header("📊 Dataset Overview")
st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
st.dataframe(df.head())

st.subheader("Missing values per column")
st.write(df.isnull().sum())

# --- VISUALIZATION 1: Distribution ---
st.header("🌍 Distribution of Ocean Health Index Scores")
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(df["OHI_index"], bins=20, kde=True, color="skyblue", ax=ax)
ax.set_title("Distribution of Ocean Health Index Scores")
st.pyplot(fig)

# --- VISUALIZATION 2: Top 10 Regions ---
if "region_name" in df.columns:
    st.header("🏆 Top 10 Regions by Ocean Health Index")
    top10 = df.nlargest(10, "OHI_index")[["region_name", "OHI_index"]]
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=top10, x="OHI_index", y="region_name", palette="viridis", ax=ax)
    ax.set_title("Top 10 Regions by Ocean Health Index")
    st.pyplot(fig)

# --- VISUALIZATION 3: Correlation Heatmap ---
st.header("🔗 Correlation Between OHI Goal Scores")
exclude_cols = ["region_name", "region_id", "OHI_index"]
num_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype != "O"]

if len(num_cols) > 1:
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df[num_cols].corr(), cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Between OHI Goal Scores")
    st.pyplot(fig)
else:
    st.info("Not enough numeric goal columns to compute correlations.")

# --- VISUALIZATION 4: OHI vs Trend ---
trend_cols = [c for c in df.columns if "trend" in c.lower()]
if trend_cols:
    st.header(f"📈 OHI Index vs {trend_cols[0]}")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.scatterplot(data=df, x="OHI_index", y=trend_cols[0], ax=ax)
    st.pyplot(fig)

# --- VISUALIZATION 5: Sub-goal Breakdown ---
st.header("🌐 Sub-goal Breakdown for a Specific Region")
if "region_name" in df.columns:
    region = st.selectbox("Choose a region:", sorted(df["region_name"].unique()))
    subset = df[df["region_name"] == region]

    if not subset.empty:
        scores = subset.iloc[0, 6:-2]
        fig, ax = plt.subplots(figsize=(10,5))
        scores.plot(kind="bar", color="teal", ax=ax)
        ax.set_title(f"{region} — Sub-Goal Scores")
        ax.set_ylabel("Score")
        st.pyplot(fig)

# --- MACHINE LEARNING ---
st.header("🤖 Machine Learning: Predicting OHI Index")

goal_columns = ['AO', 'BD', 'CP', 'CS', 'CW', 'ECO', 'FIS', 'FP', 'HAB', 'ICO',
                'LE', 'LIV', 'LSP', 'MAR', 'NP', 'SP', 'SPP', 'TR']
goal_columns = [c for c in goal_columns if c in df.columns]

if len(goal_columns) < 2:
    st.error("❌ Not enough goal columns found for training the model.")
    st.stop()

X = df[goal_columns]
y = df["OHI_index"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.write(f"**R² score:** {r2:.3f}")
st.write(f"**Mean Absolute Error:** {mae:.2f}")

importance = pd.DataFrame({
    "Goal": goal_columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

st.subheader("Top contributing goals to OHI Index")
st.dataframe(importance)

fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(data=importance, x="Coefficient", y="Goal", palette="viridis", ax=ax)
ax.set_title("Contribution of Each Goal to Ocean Health Index")
st.pyplot(fig)
