import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style setup
sns.set_theme(style="whitegrid")

# dataset inladen
file_path = "data.csv"
df = pd.read_csv(file_path)

# dataset cleanen en voorbereiden
df = df.drop(columns=['Shape__Area', 'Shape__Length'])
df = df.rename(columns={
    'rgn_nam': 'region_name',
    'Index_': 'OHI_index',
    'trnd_sc': 'trend_score'
})

# --- Visualization Section ---

# Distribution of Ocean Health Index Scores
plt.figure(figsize=(8,4))
sns.histplot(df['OHI_index'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Ocean Health Index Scores')
plt.xlabel('OHI Index')
plt.ylabel('Number of Regions')
plt.savefig("OHI_index.png", dpi=300, bbox_inches='tight')

# Top 10 Regions by Ocean Health Index
top10 = df.nlargest(10, 'OHI_index')[['region_name', 'OHI_index']]

plt.figure(figsize=(8,5))
sns.barplot(data=top10, x='OHI_index', y='region_name', palette='viridis')
plt.title('Top 10 Regions by Ocean Health Index')
plt.xlabel('OHI Index')
plt.ylabel('Region')
plt.savefig("Top10.png", dpi=300, bbox_inches='tight')

# Correlation Between OHI Goals
plt.figure(figsize=(12,8))
# Select only sub-goal columns (from AO to TR)
goal_columns = df.columns[6:-2]
sns.heatmap(df[goal_columns].corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Between OHI Goal Scores')
plt.savefig("Correlation.png", dpi=300, bbox_inches='tight')

# Relationship Between OHI Index and Trend
plt.figure(figsize=(8,4))
sns.scatterplot(data=df, x='OHI_index', y='trend_score')
plt.title('OHI Index vs Trend Score')
plt.xlabel('OHI Index')
plt.ylabel('Trend Score')
plt.savefig("OHI_vs_Trend.png", dpi=300, bbox_inches='tight')

# Sub-goal Breakdown for a Specific Region
region = "Belgium"  # Change this to explore another region
subset = df[df['region_name'] == region]

if not subset.empty:
    scores = subset.iloc[0, 6:-2]
    plt.figure(figsize=(10,5))
    scores.plot(kind='bar', color='teal')
    plt.title(f'{region} — Sub-Goal Scores')
    plt.ylabel('Score')
    plt.savefig("subset.png", dpi=300, bbox_inches='tight')
else:
    print(f"Region '{region}' not found in dataset.")
