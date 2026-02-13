import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Force dark theme
st.markdown("""
    <script>
    document.querySelector('body').style.backgroundColor = '#0e1117';
    </script>
    <style>
    /* Force Streamlit dark theme */
    .stApp {
        background-color: #0e1117 !important;
    }
    </style>
""", unsafe_allow_html=True)

CROPS = ['Bajra', 'Barley', 'Cotton', 'Groundnut', 'Jowar', 'Maize', 'Rice', 'Soybean', 'Sugarcane', 'Wheat']
SOIL_TYPES = ['Alluvial', 'Arid', 'Black', 'Forest', 'Laterite', 'Red']

MARKET_RATES = {
    'Bajra': 2500,
    'Barley': 2000,
    'Cotton': 6000,
    'Groundnut': 5500,
    'Jowar': 15000,
    'Maize': 2000,
    'Rice': 2200,
    'Soybean': 4500,
    'Sugarcane': 350,
    'Wheat': 2150
}

@st.cache_data
def load_and_train_model():
    df = pd.read_csv('data/crop_yield.csv')
    
    le_crop = LabelEncoder()
    le_soil = LabelEncoder()
    
    df['Crop_encoded'] = le_crop.fit_transform(df['Crop'])
    df['Soil_Type_encoded'] = le_soil.fit_transform(df['Soil_Type'])
    
    X = df[['Crop_encoded', 'Soil_Type_encoded', 'Rainfall_mm', 'Temperature_C', 
            'Humidity_%', 'Soil_pH', 'Nitrogen_kg/ha', 'Phosphorus_kg/ha', 'Potassium_kg/ha']]
    y = df['Yield_tonnes_per_hectare']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, le_crop, le_soil

def predict_yield(model, le_crop, le_soil, crop, soil_type, rainfall, temp, humidity, ph, n, p, k):
    crop_encoded = le_crop.transform([crop])[0]
    soil_encoded = le_soil.transform([soil_type])[0]
    
    features = np.array([[crop_encoded, soil_encoded, rainfall, temp, humidity, ph, n, p, k]])
    prediction = model.predict(features)[0]
    
    std_error = 0.15 * prediction
    confidence_low = max(0, prediction - 1.96 * std_error)
    confidence_high = prediction + 1.96 * std_error
    
    return prediction, confidence_low, confidence_high

def get_crop_recommendations(rainfall, temp, humidity, ph):
    recommendations = []
    
    if rainfall < 800:
        recommendations.extend(['Bajra', 'Cotton', 'Groundnut'])
    elif rainfall < 1500:
        recommendations.extend(['Jowar', 'Maize', 'Soybean', 'Wheat'])
    else:
        recommendations.extend(['Rice', 'Sugarcane'])
    
    if temp > 30:
        recommendations.extend(['Cotton', 'Bajra', 'Groundnut'])
    elif temp < 20:
        recommendations.extend(['Wheat', 'Barley'])
    
    recommendations = list(set(recommendations))[:3]
    return recommendations

st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        padding: 20px;
    }
    
    /* Text colors */
    .title {
        color: #4ade80;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        color: #9ca3af;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-header {
        color: #22c55e;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    /* Result boxes */
    .result-box {
        background-color: #14532d;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #22c55e;
        margin-top: 15px;
    }
    .result-box h3 {
        color: #4ade80 !important;
    }
    .result-box p {
        color: #d1d5db !important;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card div:first-child {
        color: #4ade80;
    }
    .metric-card div:nth-child(2) {
        color: #9ca3af;
    }
    .metric-card div:nth-child(3) {
        color: #6b7280;
        font-size: 14px;
        margin-top: 5px;
    }
    
    /* Warning box */
    .warning-box {
        background-color: #451a03;
        padding: 12px;
        border-radius: 8px;
        margin-top: 15px;
        color: #fed7aa;
    }
    
    /* Streamlit widget overrides */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #d1d5db !important;
    }
    .stSelectbox > div > div {
        background-color: #1f2937 !important;
        color: #f3f4f6 !important;
    }
    .stButton > button {
        background-color: #16a34a !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: #15803d !important;
    }
    
    /* Input fields */
    input[type="number"] {
        background-color: #1f2937 !important;
        color: #f3f4f6 !important;
    }
    
    /* Recommendation chips */
    .crop-chip {
        background-color: #1f2937;
        padding: 15px 25px;
        border-radius: 10px;
        font-weight: bold;
        color: #4ade80;
    }
    
    /* Footer */
    .footer {
        color: #6b7280;
        text-align: center;
        font-size: 12px;
    }
    
    /* Slider thumb */
    div[data-baseweb="slider"] > div > div > div > div {
        background-color: #22c55e !important;
    }
    
    /* Slider track */
    div[data-baseweb="slider"] > div > div > div > div[role="slider"] {
        background-color: #22c55e !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌾 Crop Yield Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your farm conditions for AI-powered predictions</div>', unsafe_allow_html=True)

try:
    model, le_crop, le_soil = load_and_train_model()
    
    st.markdown('<div class="section-header">🌾 Select Crop & Soil Type</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("Crop", CROPS, index=4)
    with col2:
        soil_type = st.selectbox("Soil Type", SOIL_TYPES, index=0)
    
    st.markdown('<div class="section-header">🌦️ Environmental Conditions</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        rainfall = st.slider("Rainfall (mm)", min_value=400, max_value=2500, value=850, step=50)
    with col4:
        temp = st.slider("Temperature (°C)", min_value=15, max_value=40, value=25, step=1)
    
    col5, col6 = st.columns(2)
    with col5:
        ph = st.slider("Soil pH", min_value=5.0, max_value=9.0, value=6.8, step=0.1, format="%.1f")
    with col6:
        humidity = st.slider("Humidity (%)", min_value=40, max_value=95, value=65, step=1)
    
    st.markdown('<div class="section-header">💊 Fertilizer Usage (Optional)</div>', unsafe_allow_html=True)
    col7, col8, col9 = st.columns(3)
    with col7:
        n = st.number_input("N (kg/ha)", min_value=0, max_value=200, value=80, step=1)
    with col8:
        p = st.number_input("P (kg/ha)", min_value=0, max_value=200, value=40, step=1)
    with col9:
        k = st.number_input("K (kg/ha)", min_value=0, max_value=200, value=50, step=1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col10, col11 = st.columns(2)
    with col10:
        predict_btn = st.button("🔮 Predict Yield", use_container_width=True)
    with col11:
        recommend_btn = st.button("💡 Get Crop Recommendations", use_container_width=True)
    
    if predict_btn:
        prediction, conf_low, conf_high = predict_yield(
            model, le_crop, le_soil, crop, soil_type, 
            rainfall, temp, humidity, ph, n, p, k
        )
        
        market_rate = MARKET_RATES[crop]
        revenue = prediction * market_rate
        
        st.markdown(f"""
            <div class="result-box">
                <h3 style="color:#4ade80; margin-top:0">✅ Prediction Results for {crop}</h3>
                <div style="display:flex; justify-content:space-around; margin:20px 0; flex-wrap:wrap; gap:10px;">
                    <div class="metric-card" style="width:30%; min-width:150px;">
                        <div style="font-size:28px; font-weight:bold; color:#4ade80">{prediction:.2f}</div>
                        <div>tonnes/hectare</div>
                        <div style="font-size:14px; color:#9ca3af; margin-top:5px">Predicted Yield</div>
                    </div>
                    <div class="metric-card" style="width:30%; min-width:150px;">
                        <div style="font-size:24px; font-weight:bold; color:#4ade80">{conf_low:.2f} - {conf_high:.2f}</div>
                        <div>tonnes/hectare</div>
                        <div style="font-size:14px; color:#9ca3af; margin-top:5px">Confidence Range</div>
                    </div>
                    <div class="metric-card" style="width:30%; min-width:150px;">
                        <div style="font-size:28px; font-weight:bold; color:#4ade80">₹{revenue:,.0f}</div>
                        <div>per hectare</div>
                        <div style="font-size:14px; color:#9ca3af; margin-top:5px">Expected Revenue</div>
                    </div>
                </div>
                <p style="color:#d1d5db; font-style:italic; margin-top:15px">
                    💡 Based on market rate of ₹{market_rate:,}/tonne for {crop}. 
                    Actual prices vary by quality and local market conditions.
                </p>
            </div>
            <div class="warning-box">
                <strong>⚠️ Important:</strong> This is an AI estimate. Actual yields depend on pests, diseases, 
                local conditions, and farming practices. Please consult agricultural experts for critical decisions.
            </div>
        """, unsafe_allow_html=True)
    
    if recommend_btn:
        recommendations = get_crop_recommendations(rainfall, temp, humidity, ph)
        
        st.markdown(f"""
            <div class="result-box">
                <h3 style="color:#4ade80; margin-top:0">💡 Recommended Crops</h3>
                <p style="color:#d1d5db;">Based on your current conditions (Rainfall: {rainfall}mm, Temperature: {temp}°C, Humidity: {humidity}%, pH: {ph}):</p>
                <div style="display:flex; justify-content:center; gap:15px; flex-wrap:wrap; margin:20px 0;">
                    {''.join([f'<div class="crop-chip">{r}</div>' for r in recommendations])}
                </div>
                <p style="color:#d1d5db; font-style:italic;">These crops are best suited for your environmental conditions.</p>
            </div>
        """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("⚠️ Error: Could not find 'data/crop_yield.csv'. Please ensure the data file exists in the 'data' folder.")
except Exception as e:
    st.error(f"⚠️ An error occurred: {str(e)}")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#6b7280; font-size:12px;'>Crop Yield Prediction System | AICTE MS Elevate Internship | January 2026</p>", unsafe_allow_html=True)
