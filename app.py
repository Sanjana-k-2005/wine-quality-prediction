import streamlit as st
import pandas as pd
import joblib

# ---------- Page config ----------
st.set_page_config(page_title="Wine Predictor", page_icon="🍷")

# ---------- Load model ----------
@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")

model = load_model()

# ---------- Title ----------
st.title("🍷 Wine Quality Predictor")

# ---------- Inputs (simple layout) ----------
st.subheader("Enter Wine Details")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed acidity", value=8.3)
    volatile_acidity = st.number_input("Volatile acidity", value=0.53)
    citric_acid = st.number_input("Citric acid", value=0.27)
    residual_sugar = st.number_input("Residual sugar", value=2.5)
    chlorides = st.number_input("Chlorides", value=0.087)
    free_sulfur_dioxide = st.number_input("Free sulfur dioxide", value=15.0)

with col2:
    total_sulfur_dioxide = st.number_input("Total sulfur dioxide", value=46.0)
    density = st.number_input("Density", value=0.9967, format="%.4f")
    pH = st.number_input("pH", value=3.31)
    sulphates = st.number_input("Sulphates", value=0.66)
    alcohol = st.number_input("Alcohol", value=10.4)

# ---------- Data ----------
input_df = pd.DataFrame([{
    "fixed acidity": fixed_acidity,
    "volatile acidity": volatile_acidity,
    "citric acid": citric_acid,
    "residual sugar": residual_sugar,
    "chlorides": chlorides,
    "free sulfur dioxide": free_sulfur_dioxide,
    "total sulfur dioxide": total_sulfur_dioxide,
    "density": density,
    "pH": pH,
    "sulphates": sulphates,
    "alcohol": alcohol,
}])

# ---------- Predict ----------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    if str(prediction).lower() == "good":
        st.success(f"Good Quality 🍇")
    else:
        st.error(f"Bad Quality 🧪")