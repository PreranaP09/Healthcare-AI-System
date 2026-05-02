%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Visualization

import seaborn as sns
import matplotlib.pyplot as plt

# -------------------- PAGE CONFIG --------------------

st.set_page_config(page_title="Disease Risk Prediction", layout="wide")

# -------------------- LOAD FILES --------------------

lr = pickle.load(open('model.pkl', 'rb'))
kmeans = pickle.load(open('kmeans.pkl', 'rb'))

scaler = pickle.load(open('scaler.pkl', 'rb'))
cluster_scaler = pickle.load(open('cluster_scaler.pkl', 'rb'))

X_columns = pickle.load(open('X_columns.pkl', 'rb'))
cluster_features = pickle.load(open('cluster_features.pkl', 'rb'))

df = pd.read_csv("/content/clean_healthcare_data.csv")

# -------------------- SIDEBAR --------------------

st.sidebar.title("🧠 Healthcare AI System")
menu = st.sidebar.radio("Navigation", ["🏠 Prediction", "📊 EDA", "📈 Model Performance"])

# =====================================================

# 🏠 PREDICTION PAGE

# =====================================================

if menu == "🏠 Prediction":

  st.title("🩺 Disease Risk Prediction")

  col1, col2 = st.columns(2)

  with col1:
      age = st.number_input("Age", min_value=0, step=1)
      # age = st.number_input("Age", min_value=0.0)
      bmi = st.number_input("BMI")
      bp = st.number_input("Blood Pressure")

  with col2:
      chol = st.number_input("Cholesterol Level")
      gluc = st.number_input("Glucose Level")

  if st.button("Predict"):

      # ---------------- FULL INPUT ----------------
      final_input = pd.DataFrame(0, index=[0], columns=X_columns)

      user_dict = {
          'Age': age,
          'BMI': bmi,
          'Blood_Pressure': bp,
          'Cholesterol_Level': chol,
          'Glucose_Level': gluc
      }

      for col in user_dict:
          if col in final_input.columns:
              final_input[col] = user_dict[col]

      # ---------------- FEATURE ENGINEERING ----------------
      if 'Age_Glucose' in final_input.columns:
          final_input['Age_Glucose'] = final_input['Age'] * final_input['Glucose_Level']

      if 'BP_Chol_Risk' in final_input.columns:
          final_input['BP_Chol_Risk'] = final_input['Blood_Pressure'] * final_input['Cholesterol_Level']

      # Ensure correct order
      final_input = final_input[X_columns]

      # ---------------- SCALE + PREDICT ----------------
      final_scaled = scaler.transform(final_input)
      risk = lr.predict(final_scaled)[0]

      # ---------------- RISK LEVEL ----------------
      # if risk < 0.3:
      #     level = "🟢 Low"
      # elif risk < 0.6:
      #     level = "🟡 Medium"
      # else:
      #     level = "🔴 High"

      # ================= CLUSTER =================
      cluster_input = pd.DataFrame([{
          'Glucose_Level': gluc,
          'BMI': bmi,
          'Age_Glucose': age * gluc
      }])

      cluster_input = cluster_input[cluster_features]
      cluster_scaled = cluster_scaler.transform(cluster_input)
      cluster = kmeans.predict(cluster_scaled)[0]

      # ---------------- CLUSTER LABEL ----------------
      cluster_labels = {
          0: 'Critical Risk',
          1: 'High Glucose Pattern',
          2: 'Obesity Pattern',
          3: 'Low Risk Lifestyle'
      }

      cluster_name = cluster_labels.get(cluster, "Unknown")

      # ---------------- OUTPUT ----------------
      st.subheader("📌 Results")

      colA, colB, colC = st.columns(3)

      colA.metric("Risk Score", round(risk, 3))
      # colB.metric("Risk Level", level)
      colB.metric("Cluster", int(cluster))
      colC.markdown(f"### 🧬 Cluster Type: **{cluster_name}**")
      # colC.markdown("Cluster Type", cluster_name)

      # ⭐ Cluster Type Display
      # st.markdown(f"### 🧬 Cluster Type: **{cluster_name}**")

      # ---------------- PROGRESS BAR ----------------
      st.progress(float(risk))

      # ---------------- USER VISUAL ----------------
      st.subheader("📊 Your Health Overview")
      col1, col2, col3 = st.columns([1,2,1])

      with col2:
          fig, ax = plt.subplots(figsize=(6, 6))

          pd.DataFrame(user_dict, index=["You"]).T.plot(kind='bar', ax=ax, legend=False)

          plt.xticks(rotation=45)
          plt.tight_layout()

          st.pyplot(fig)

# =====================================================

# 📊 EDA PAGE

# =====================================================

elif menu == "📊 EDA":

  st.title("📊 Exploratory Data Analysis")

  st.subheader("Dataset Preview")
  st.dataframe(df.head())

  st.subheader("📉 Data Distribution")
  numeric_cols = df.select_dtypes(include=['number']).columns

  for i in range(0, len(numeric_cols), 2):
      cols = st.columns(2)

      for j in range(2):
          if i + j < len(numeric_cols):
              col_name = numeric_cols[i + j]

              with cols[j]:
                  fig, ax = plt.subplots(figsize=(5, 4))

                  sns.histplot(df[col_name], kde=True, ax=ax)

                  # ---------------- SKEWNESS ----------------
                  skew_val = df[col_name].skew()

                  if skew_val > 0.5:
                      skew_type = "Right Skewed"
                  elif skew_val < -0.5:
                      skew_type = "Left Skewed"
                  else:
                      skew_type = "Approximately Symmetric"

                  # Title with skew info
                  ax.set_title(f"{col_name}\nSkew: {round(skew_val,2)} ({skew_type})")

                  st.pyplot(fig)

  st.subheader("📦 Outlier Detection")

  num_out = [
    'Age', 'BMI', 'Blood_Pressure',
    'Cholesterol_Level', 'Glucose_Level',
    'Age_Glucose', 'BP_Chol_Risk'
      ]

  for i in range(0, len(num_out), 2):  # 2 per row
      cols = st.columns(2)

      for j in range(2):
          if i + j < len(num_out):
              col_name = num_out[i + j]

              with cols[j]:
                  fig, ax = plt.subplots(figsize=(5, 4))

                  sns.boxplot(y=df[col_name], ax=ax)

                  ax.set_title(col_name.replace("_", " "))

                  st.pyplot(fig)

# =====================================================

# 📈 MODEL PERFORMANCE

# =====================================================

elif menu == "📈 Model Performance":

    st.title("📈 Model Performance")

    # ---------------- LOAD DATA ----------------
    raw_df = pd.read_csv("/content/clean_healthcare_data.csv")

    # ---------------- TARGET ----------------
    y_actual = raw_df['Disease_Risk_Score']

    # ---------------- FEATURE DATA ----------------
    df_model = raw_df.copy()

    # Feature engineering
    df_model['Age_Glucose'] = df_model['Age'] * df_model['Glucose_Level']
    df_model['BP_Chol_Risk'] = df_model['Blood_Pressure'] * df_model['Cholesterol_Level']

    # Encoding
    df_model = pd.get_dummies(df_model)

    # Add missing columns
    for col in X_columns:
        if col not in df_model.columns:
            df_model[col] = 0

    # Ensure correct order
    df_model = df_model[X_columns]
    df_model = df_model.fillna(0)

    # ---------------- SCALE + PREDICT ----------------
    X_scaled = scaler.transform(df_model)
    y_pred = lr.predict(X_scaled)

    # ---------------- REMOVE NaN ----------------
    mask = y_actual.notna()
    y_actual = y_actual[mask]
    y_pred = y_pred[mask]

    # ---------------- SELECT X AXIS ----------------
    x = raw_df.loc[mask, 'Glucose_Level']

    # ---------------- BEST FIT LINE ----------------
    m, b = np.polyfit(x, y_pred, 1)

    # Sort for smooth line
    sorted_idx = np.argsort(x)
    x_sorted = x.iloc[sorted_idx]
    y_line = m * x_sorted + b

    # ---------------- PLOT ----------------
    col1, col2, col3 = st.columns([1, 2, 1])  # middle column bigger

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))

        # Actual (blue)
        ax.scatter(x, y_actual, label="Actual Risk", alpha=0.6)

        # Predicted (green)
        ax.scatter(x, y_pred, label="Predicted Risk", alpha=0.6)

        # Best fit (red)
        ax.plot(x_sorted, y_line, linewidth=2, label="Best Fit Line")

        ax.set_xlabel("Glucose Level")
        ax.set_ylabel("Disease Risk Score")
        ax.set_title("Glucose vs Disease Risk Score")

        ax.legend()

        st.pyplot(fig)

    # ---------------- METRICS ----------------
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

    st.write("MAE:", round(mae, 4))
    st.write("RMSE:", round(rmse, 4))
