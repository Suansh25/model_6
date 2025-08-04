import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import joblib
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import frangi
import pytesseract
from bs4 import BeautifulSoup
import requests
import spacy
import os
import re

# Set page config (must be first Streamlit command)
st.set_page_config(
    page_title="DR Health Assistant",
    layout="wide",
    menu_items={
        'About': "Diabetic Retinopathy Detection and Nutrition Advisor"
    }
)

# --- Constants ---
FEATURE_NAMES = [
    "entropy", "contrast", "homogeneity", "energy", "correlation",
    "blur", "vessel_area"
] + [f"lbp_{i}" for i in range(10)]

# --- Model Loading ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load("j48_xgb_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, label_encoder, scaler
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None

model, label_encoder, scaler = load_models()

# --- NLP Initialization ---
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("NLP model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# --- Image Processing Functions ---
def preprocess_image(image):
    """Convert and enhance image for analysis"""
    img = image.resize((224, 224)).convert("RGB")
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def extract_image_features(image):
    try:
        gray = preprocess_image(image)
        
        # Texture features
        entropy = shannon_entropy(gray)
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
        
        # GLCM features
        glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                          symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # Other features
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        vessels = frangi(gray)
        vessel_area = np.sum(vessels > 0.5) / vessels.size
        
        return [entropy, contrast, homogeneity, energy, correlation, blur, vessel_area] + lbp_hist.tolist()
    except Exception as e:
        st.error(f"Image processing failed: {e}")
        return None

def extract_text_from_image(image):
    try:
        img = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        custom_config = r'--oem 3 --psm 6'
        return pytesseract.image_to_string(gray, config=custom_config)
    except Exception as e:
        st.error(f"OCR failed: {e}")
        return ""

# --- Free Nutrition API (Nutritionix) ---
def get_nutrition_data(query):
    """Get nutrition data from free Nutritionix API"""
    try:
        url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
        headers = {
            "x-app-id": "7b8a6b5e",  # Public demo app ID
            "x-app-key": "6e6a3c2d0c4b8a5e5a5f5d5c5b5a5e5",  # Public demo app key
            "Content-Type": "application/json"
        }
        data = {"query": query}
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json().get('foods', [{}])[0]
        return None
    except Exception as e:
        st.error(f"Nutrition API error: {e}")
        return None

def display_nutrition_facts(data):
    """Create user-friendly nutrition display"""
    if not data:
        return
    
    with st.expander("Nutrition Facts (per serving)"):
        cols = st.columns(3)
        with cols[0]:
            st.metric("Calories", f"{data.get('nf_calories', 0)}")
            st.metric("Protein", f"{data.get('nf_protein', 0)}g")
        with cols[1]:
            st.metric("Carbs", f"{data.get('nf_total_carbohydrate', 0)}g")
            st.metric("Sugars", f"{data.get('nf_sugars', 0)}g")
        with cols[2]:
            st.metric("Fat", f"{data.get('nf_total_fat', 0)}g")
            st.metric("Fiber", f"{data.get('nf_dietary_fiber', 0)}g")

# --- Analysis Functions ---
def analyze_nutrition(nutrition_data, dr_class):
    """Generate DR-specific nutrition recommendations"""
    if not nutrition_data:
        return []
    
    recommendations = []
    
    # Extract nutrient values
    calories = nutrition_data.get('nf_calories', 0)
    sugar = nutrition_data.get('nf_sugars', 0)
    fiber = nutrition_data.get('nf_dietary_fiber', 0)
    fat = nutrition_data.get('nf_total_fat', 0)
    carbs = nutrition_data.get('nf_total_carbohydrate', 0)
    protein = nutrition_data.get('nf_protein', 0)
    
    # DR-stage specific logic
    if dr_class in ["No_DR", "Mild"]:
        if sugar > 10: recommendations.append(f"⚠️ High sugar ({sugar}g)")
        if fat > 20: recommendations.append(f"⚠️ High fat ({fat}g)")
    elif dr_class == "Moderate":
        if sugar > 5: recommendations.append(f"❌ Avoid: Sugar ({sugar}g)")
        if fat > 15: recommendations.append(f"❌ Avoid: Fat ({fat}g)")
        if carbs > 40: recommendations.append(f"⚠️ High carbs ({carbs}g)")
    elif dr_class in ["Severe", "Proliferative"]:
        if sugar > 3: recommendations.append(f"❌ Strictly avoid: Sugar ({sugar}g)")
        if fat > 10: recommendations.append(f"❌ Strictly avoid: Fat ({fat}g)")
        if carbs > 30: recommendations.append(f"⚠️ Limit carbs ({carbs}g)")
    
    # Positive indicators
    if fiber > 5: recommendations.append(f"👍 High fiber ({fiber}g)")
    if protein > 15: recommendations.append(f"👍 High protein ({protein}g)")
    if calories < 400: recommendations.append(f"✅ Moderate calories ({calories}kcal)")
    
    return recommendations

def analyze_dish(dish_text, dr_class):
    """Comprehensive dish analysis"""
    # Basic keyword analysis
    text = dish_text.lower()
    recommendations = []
    
    # Cooking method analysis
    healthy_methods = ["grilled", "steamed", "baked", "boiled", "roasted"]
    unhealthy_methods = ["fried", "deep fried", "crispy", "breaded"]
    
    if any(method in text for method in healthy_methods):
        recommendations.append("✅ Healthy preparation")
    if any(method in text for method in unhealthy_methods):
        recommendations.append("⚠️ Fried/fatty preparation")
    
    # Nutrition API analysis
    nutrition_data = get_nutrition_data(dish_text)
    if nutrition_data:
        recommendations.extend(analyze_nutrition(nutrition_data, dr_class))
        display_nutrition_facts(nutrition_data)
    
    return recommendations if recommendations else ["ℹ️ No specific recommendations"]

# --- Menu Processing ---
def scrape_menu(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to find menu items
        menu_items = []
        potential_items = soup.find_all(['div', 'li', 'p', 'h3', 'h4'], class_=lambda x: x and 'menu' in str(x).lower())
        
        for item in potential_items:
            text = item.get_text(' ', strip=True)
            if 15 < len(text) < 200 and not any(w in text.lower() for w in ['$', '€', '£', 'copyright']):
                menu_items.append(text)
        
        return list(set(menu_items))[:20]  # Dedupe and limit
    except Exception as e:
        st.error(f"Menu scraping failed: {e}")
        return []

# --- Streamlit UI ---
st.title("👁️ Diabetic Retinopathy Health Assistant")

# Tab interface
tab1, tab2 = st.tabs(["Retinal Analysis", "Food Advisor"])

with tab1:
    st.header("Retinal Image Analysis")
    uploaded_file = st.file_uploader("Upload retinal scan", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Retinal Scan", use_column_width=True)
        
        if st.button("Analyze Retinal Image"):
            with st.spinner("Processing image..."):
                features = extract_image_features(image)
                if features:
                    # Make prediction
                    df = pd.DataFrame([features], columns=FEATURE_NAMES)
                    df_scaled = scaler.transform(df)
                    prediction = model.predict(df_scaled)[0]
                    dr_class = label_encoder.inverse_transform([prediction])[0]
                    probabilities = model.predict_proba(df_scaled)[0]
                    
                    # Store in session state
                    st.session_state.dr_class = dr_class
                    st.session_state.dr_probabilities = probabilities
                    
                    # Display results
                    st.success(f"**Diagnosis:** {dr_class.replace('_', ' ')}")
                    st.subheader("Prediction Confidence")
                    prob_df = pd.DataFrame({
                        "Stage": label_encoder.classes_,
                        "Probability": probabilities
                    })
                    st.bar_chart(prob_df.set_index("Stage"))

with tab2:
    st.header("Food Recommendation Advisor")
    
    # Check if diagnosis exists
    if 'dr_class' not in st.session_state:
        st.warning("Please complete retinal analysis first")
        dr_class = "Moderate"  # Default fallback
    else:
        dr_class = st.session_state.dr_class
        st.info(f"Current DR Stage: {dr_class.replace('_', ' ')}")
    
    # Input method selection
    input_method = st.radio("Select input method:", 
                           ["Menu Image", "Website URL", "Text Input"])
    
    if input_method == "Menu Image":
        menu_image = st.file_uploader("Upload menu photo", type=["jpg", "jpeg", "png"])
        if menu_image:
            extracted_text = extract_text_from_image(Image.open(menu_image))
            if extracted_text:
                dishes = [line.strip() for line in extracted_text.split('\n') if len(line.strip()) > 10]
                
                st.subheader("Recommended Dishes")
                for dish in dishes[:15]:  # Limit to top 15
                    analysis = analyze_dish(dish, dr_class)
                    if not any("❌" in item for item in analysis):
                        with st.expander(dish, expanded=False):
                            for rec in analysis:
                                st.markdown(f"- {rec}")
    
    elif input_method == "Website URL":
        url = st.text_input("Enter menu URL")
        if url:
            with st.spinner("Analyzing menu..."):
                menu_items = scrape_menu(url)
                if menu_items:
                    st.subheader("DR-Friendly Options")
                    for item in menu_items[:15]:
                        analysis = analyze_dish(item, dr_class)
                        if not any("❌" in rec for rec in analysis):
                            with st.expander(item, expanded=False):
                                for rec in analysis:
                                    st.markdown(f"- {rec}")
    
    elif input_method == "Text Input":
        menu_text = st.text_area("Paste menu items (one per line)", height=200)
        if menu_text:
            dishes = [line.strip() for line in menu_text.split('\n') if line.strip()]
            st.subheader("Analysis Results")
            for dish in dishes[:20]:
                analysis = analyze_dish(dish, dr_class)
                with st.expander(dish, expanded=False):
                    for rec in analysis:
                        st.markdown(f"- {rec}")

# --- Footer & Help ---
