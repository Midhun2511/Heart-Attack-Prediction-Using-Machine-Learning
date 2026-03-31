import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Dashboard",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.main {background-color: #0e1117;}
h1, h2, h3 {color: white;}
.stMetric {background-color: #1c1f26; padding: 15px; border-radius: 10px;}
div.stButton > button {
    background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# -----------------------------
# Title
# -----------------------------
st.title("❤️ Heart Disease Prediction Dashboard")
st.caption("AI-Based Health Risk Analysis System")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.title("🧾 Patient Details")

# 🧍 Basic Info
st.sidebar.markdown("### 🧍 Basic Information")
age = st.sidebar.number_input("🎂 Age", 1, 120, 30)
sex = st.sidebar.selectbox("🧑 Gender", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

# ❤️ Health Metrics
st.sidebar.markdown("### ❤️ Health Metrics")
chol = st.sidebar.number_input("🧪 Cholesterol", 100, 500, 200)
bp = st.sidebar.selectbox("🩸 Blood Pressure", ["Normal", "High", "Very High"])
bp = {"Normal": 0, "High": 1, "Very High": 2}[bp]
heart_rate = st.sidebar.number_input("❤️ Heart Rate", 40, 220, 80)
bmi = st.sidebar.number_input("⚖️ BMI", 10.0, 50.0, 22.0)
trig = st.sidebar.selectbox("🧬 Triglycerides", ["Normal", "Borderline", "High"])
trig = {"Normal": 0, "Borderline": 1, "High": 2}[trig]

# 🩺 Conditions
st.sidebar.markdown("### 🩺 Medical Conditions")
diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
diabetes = 1 if diabetes == "Yes" else 0
family = st.sidebar.selectbox("Family History", ["No", "Yes"])
family = 1 if family == "Yes" else 0
previous = st.sidebar.selectbox("Previous Heart Problems", ["No", "Yes"])
previous = 1 if previous == "Yes" else 0
medication = st.sidebar.selectbox("💊 Medication Use", ["No", "Yes"])
medication = 1 if medication == "Yes" else 0

# 🏃 Lifestyle
st.sidebar.markdown("### 🏃 Lifestyle")
smoking = st.sidebar.selectbox("🚬 Smoking", ["No", "Yes"])
smoking = 1 if smoking == "Yes" else 0
obesity = st.sidebar.selectbox("⚖️ Obesity", ["No", "Yes"])
obesity = 1 if obesity == "Yes" else 0
alcohol = st.sidebar.selectbox("🍺 Alcohol", ["None", "Low", "Moderate", "High"])
alcohol = {"None": 0, "Low": 1, "Moderate": 2, "High": 3}[alcohol]
exercise = st.sidebar.slider("🏃 Exercise Hours/Week", 0, 20, 3)
activity = st.sidebar.slider("🏃 Activity Days/Week", 0, 7, 3)
sleep = st.sidebar.slider("😴 Sleep Hours/Day", 0, 12, 7)
stress = st.sidebar.slider("😓 Stress Level", 0, 10, 5)
sedentary = st.sidebar.slider("🪑 Sedentary Hours/Day", 0, 15, 6)

# 🍽 Extra
st.sidebar.markdown("### 🍽 Additional Factors")
diet = st.sidebar.selectbox("🥗 Diet", ["Poor", "Average", "Good"])
diet = {"Poor": 0, "Average": 1, "Good": 2}[diet]
income = st.sidebar.number_input("💰 Income", 0, 1000000, 50000)

# 🌍 Location
st.sidebar.markdown("### 🌍 Location")
countries = ["India", "USA", "UK", "Germany", "Canada", "Australia"]
country = st.sidebar.selectbox("Country", countries)
country = countries.index(country)

continents = ["Asia", "Europe", "North America", "South America", "Africa", "Australia"]
continent = st.sidebar.selectbox("Continent", continents)
continent = continents.index(continent)

hemispheres = ["Northern", "Southern"]
hemisphere = st.sidebar.selectbox("Hemisphere", hemispheres)
hemisphere = hemispheres.index(hemisphere)

# -----------------------------
# DataFrame
# -----------------------------
data = {
    "Age": age,
    "Sex": sex,
    "Cholesterol": chol,
    "Blood Pressure": bp,
    "Heart Rate": heart_rate,
    "BMI": bmi,
    "Triglycerides": trig,
    "Diabetes": diabetes,
    "Family History": family,
    "Previous Heart Problems": previous,
    "Medication Use": medication,
    "Smoking": smoking,
    "Obesity": obesity,
    "Alcohol Consumption": alcohol,
    "Exercise Hours Per Week": exercise,
    "Physical Activity Days Per Week": activity,
    "Sleep Hours Per Day": sleep,
    "Stress Level": stress,
    "Sedentary Hours Per Day": sedentary,
    "Diet": diet,
    "Income": income,
    "Country": country,
    "Continent": continent,
    "Hemisphere": hemisphere
}

df = pd.DataFrame([data])

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1.2, 1.8])

with col1:
    st.subheader("📊 Patient Data")
    st.dataframe(df, use_container_width=True)

with col2:
    st.subheader("📈 Prediction Result")

    if st.button("🚀 Predict Risk"):

        # -----------------------------
        # AUTO FIX FEATURE MISMATCH
        # -----------------------------
        try:
            expected_columns = scaler.feature_names_in_
        except:
            expected_columns = model.feature_names_in_

        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]

        # -----------------------------
        # Prediction
        # -----------------------------
        scaled = scaler.transform(df)
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1] * 100

        c1, c2 = st.columns(2)
        c1.metric("Risk Score", f"{prob:.2f}%")
        c2.metric("Status", "High Risk" if pred else "Low Risk")

        if pred:
            st.error("⚠️ High Risk Detected")
        else:
            st.success("✅ Low Risk")

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': "Heart Risk"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"},
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Feature Importance (FIXED)
# -----------------------------
st.markdown("---")
st.subheader("📌 Feature Importance")

if hasattr(model, "feature_importances_"):

    imp_df = pd.DataFrame({
        "Feature": df.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

elif hasattr(model, "coef_"):

    coef = model.coef_[0]

    imp_df = pd.DataFrame({
        "Feature": df.columns,
        "Importance": coef
    })

    imp_df["Abs"] = imp_df["Importance"].abs()
    imp_df = imp_df.sort_values(by="Abs", ascending=False)

    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Impact (Positive ↑ Risk, Negative ↓ Risk)"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("⚠️ Model does not support feature importance")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("💡 Machine Learning + Streamlit Healthcare Dashboard")