import streamlit as st
import joblib
from PIL import Image
import base64
import os

# 🏥 Set Page Configuration
st.set_page_config(page_title="Disease Prediction System", page_icon="🩺", layout="wide")

# 🔹 Function to Set Background Image Properly
def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            base64_img = base64.b64encode(img_file.read()).decode()
        page_bg = f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{base64_img}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """
        st.markdown(page_bg, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Background image not found. Please check the file path.")

# ✅ Ensure the correct image path
set_background("assets/background.jpg")  # Make sure this image exists!

# 🎨 Custom Styling for Title
st.markdown("""
    <style>
    .title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        color: white;
        background-color: #004aad;
        padding: 10px 20px;
        border-radius: 15px;
        display: block;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 🏥 Title & Image Side-by-Side Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<h1 class="title">🩺 Disease Prediction System</h1>', unsafe_allow_html=True)

with col2:
    doctor_img_path = "assets/pngwing.com.png"  # Ensure correct filename
    if os.path.exists(doctor_img_path):
        doctor_img = Image.open(doctor_img_path)
        st.image(doctor_img, width=250)
    else:
        st.warning("⚠️ Doctor image not found.")

# 🔍 Load Model and Vectorizer
try:
    model = joblib.load("models/disease_prediction_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e}")
    st.stop()

# 📝 User Input Section
st.write("### Describe your symptoms to predict the possible disease:")
user_input = st.text_area("Enter symptoms here:", "")

# 🔍 Predict Disease Button
if st.button("🔍 Predict Disease", key="predict", help="Click to predict disease"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter your symptoms.")
    else:
        try:
            # ✅ Ensure predicted_disease is always assigned
            input_data = vectorizer.transform([user_input])
            prediction = model.predict(input_data)

            if len(prediction) > 0:  
                predicted_disease = label_encoder.inverse_transform(prediction)[0]
            else:
                predicted_disease = "Unknown Disease"

            # 🎨 Display Result in Dark Grey Box
            st.markdown(
                f"""
                <div style="
                    background-color: #333333;
                    padding: 15px;
                    border-radius: 10px;
                    color: white;
                    font-size: 18px;
                    font-weight: bold;
                    text-align: center;">
                    🩺 Predicted Disease: {predicted_disease}
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"❌ Error in prediction: {e}")
