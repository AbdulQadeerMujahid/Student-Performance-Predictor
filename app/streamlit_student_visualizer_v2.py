import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("🎓 Student Pass/Fail Prediction App")
st.markdown("""
Analyze student performance using **Previous Grades**, **Attendance Rate**, and **Study Hours per Week**  
to predict whether a student will **Pass or Fail**.
""")

# ------------------ LOAD DATA ------------------
file_path = "dataset/student_performance_prediction.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    st.success(f"✅ Loaded dataset: {file_path}")
else:
    st.error("❌ Dataset not found! Please place 'student_performance_prediction.csv' in this app’s folder.")
    st.stop()

# ------------------ DATA CLEANING ------------------
st.subheader("🧹 Data Cleaning & Preparation")

expected_cols = ["Previous Grades", "Attendance Rate", "Study Hours per Week"]
available_cols = [col for col in expected_cols if col in df.columns]

if len(available_cols) < 3:
    st.error(f"Dataset missing required columns! Found only: {available_cols}")
    st.stop()

df = df[available_cols].copy()

# Handle missing and outlier data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df = df[(df["Previous Grades"] >= 0) & (df["Previous Grades"] <= 100)]
df = df[(df["Attendance Rate"] >= 0) & (df["Attendance Rate"] <= 100)]
df = df[df["Study Hours per Week"] >= 0]

# Create target variable
df["Pass/Fail"] = np.where(
    (df["Previous Grades"] >= 50) & (df["Attendance Rate"] >= 70),
    "Pass",
    "Fail"
)

st.dataframe(df.head())

# ------------------ VISUALIZATION ------------------
st.subheader("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)
with col1:
    st.write("**Distribution of Features**")
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    sns.histplot(df["Previous Grades"], kde=True, ax=ax[0], color="skyblue")
    sns.histplot(df["Attendance Rate"], kde=True, ax=ax[1], color="salmon")
    sns.histplot(df["Study Hours per Week"], kde=True, ax=ax[2], color="lightgreen")
    ax[0].set_title("Previous Grades")
    ax[1].set_title("Attendance Rate")
    ax[2].set_title("Study Hours/Week")
    st.pyplot(fig)

with col2:
    st.write("**Correlation Heatmap**")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ------------------ PIE CHART ------------------
st.subheader("🥧 Pass/Fail Distribution")

pf_counts = df["Pass/Fail"].value_counts()
fig, ax = plt.subplots()
ax.pie(
    pf_counts,
    labels=pf_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["tomato", "lightgreen"]
)
ax.axis("equal")
st.pyplot(fig)

# ------------------ MACHINE LEARNING ------------------
st.subheader("🤖 Machine Learning Models")

X = df[["Previous Grades", "Attendance Rate", "Study Hours per Week"]]
y = df["Pass/Fail"].map({"Pass": 1, "Fail": 0})  # binary encoding

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree (ID3)": DecisionTreeClassifier(criterion="entropy", random_state=42),
    "Naive Bayes": GaussianNB()
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

result_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
st.dataframe(result_df.style.highlight_max(axis=0, color="lightgreen"))

# ------------------ CONFUSION MATRIX ------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
preds = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, preds)

st.markdown(f"### 🏆 Best Model: **{best_model_name}**")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

st.text("Classification Report:")
st.text(classification_report(y_test, preds))

# ------------------ CLUSTERING ------------------
st.subheader("🌀 K-Means Clustering (Unsupervised Insights)")

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

fig, ax = plt.subplots()
sns.scatterplot(
    x="Previous Grades",
    y="Attendance Rate",
    hue="Cluster",
    data=df,
    palette="coolwarm",
    ax=ax
)
plt.title("K-Means Clustering Results")
st.pyplot(fig)

st.success("✅ Analysis complete! Scroll up for EDA and ML insights.")
