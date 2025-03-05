import os
import json
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re
import pandas as pd
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image
import nltk

# 🔹 Pastikan nltk data disimpan di direktori yang benar
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# 🔹 Download resources yang diperlukan
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# 🔹 Load Google Cloud Credentials dari Hugging Face Secrets
if "GCP_KEY" in st.secrets:
    try:
        gcp_credentials = json.loads(st.secrets["GCP_KEY"])  # Load JSON dari Secrets
        credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
        client = vision.ImageAnnotatorClient(credentials=credentials)  # Buat klien Vision API
    except Exception as e:
        st.error(f"❌ Failed to load Google Cloud credentials: {e}")
        st.stop()
else:
    st.error("❌ GCP_KEY not found in secrets. Add in Hugging Face Settings!")
    st.stop()

# 🔹 Load Model & Tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = tf.keras.models.load_model('halal_haram_lstm_finetuned_model.h5')
        with open('tokenizer.pkl', 'rb') as file:
            tokenizer = pickle.load(file)
        return model, tokenizer
    except Exception as e:
        st.error(f"⚠ Error loading model/tokenizer: {e}")
        return None, None

model, tokenizer = load_model_and_tokenizer()

if model is None or tokenizer is None:
    st.error("⚠ Model or Tokenizer failed to load. Please double check the required files..")
    st.stop()

# 🔹 Load Data Bahan Meragukan (Kode E)
@st.cache_data
def load_kode_e_list():
    try:
        df_kode_e = pd.read_csv("kode_e_halal_check.csv")
        if 'Nama Bahan' not in df_kode_e.columns:
            st.error("⚠ CSV does not have a 'Material Name' column. Double check the file format.!")
            return set()
        return set(df_kode_e['Nama Bahan'].dropna().str.lower())
    except Exception as e:
        st.error(f"⚠ Failed to load list of questionable ingredients: {e}")
        return set()

kode_e_list = load_kode_e_list()

# 🔹 Inisialisasi NLP
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# 🔹 Stopwords Tambahan untuk Menghapus Bagian Non-Komposisi
irrelevant_words = [
    "Emulsifier","Emulsifiers","customer service", "mailbox", "po box", "contact", "website", "email", "barcode",
    "address", "phone", "manufactured", "nutrition", "produced in", "for more info", "www",
    "distributor", "net weight", "ingredients may contain", "shelf life", "tel", "fax",
    "consumer inquiries", "see", "directions", "warnings", "expiry date", "batch number",
    "CUSTOMER SER","Mail Box", "Hotline", "Email :", "PO BOXER :+62-295"
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi ekstra

    # 🔹 Hapus kata-kata tidak relevan lebih ketat
    text = re.sub(r'\b(?:' + '|'.join(irrelevant_words) + r')\b', '', text, flags=re.IGNORECASE)

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# 🔹 Cek bahan mencurigakan dalam komposisi
def check_suspicious_ingredients(text):
    text = text.lower()
    found_ingredients = [bahan for bahan in kode_e_list if bahan in text]
    return list(set(found_ingredients))  # Hapus duplikasi

# 🔹 Prediksi Halal atau Haram
def predict_label(text):
    text_clean = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text_clean])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')
    prediction = model.predict(padded_sequence)[0][0]
    confidence = round(prediction * 100, 2)

    label = "Halal" if confidence >= 50 else "Haram"
    warning_ingredients = check_suspicious_ingredients(text)
    return label, confidence, warning_ingredients

# 🔹 Ekstraksi teks dari gambar dengan OCR
def extract_text_from_image(image):
    content = image.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        return None

    full_text = texts[0].description  # Hasil OCR penuh

    # 🔹 Regex hanya untuk Ingredients
    pattern = r"(?:Ingredients)[:\s](.*?)(?:\n\n|\ncontains|\nallergen|\nnutrition|\ndistributed|\nproduced|\nmanufactured|\nmay contain|\nexpiry|\nbatch|\nwarning|\nsee packaging|$)"
    match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)

    if match:
        extracted_text = match.group(1).strip()

        # 🔹 Ambil hanya 6 baris pertama
        extracted_lines = extracted_text.split("\n")[:10]
        extracted_text = " ".join(extracted_lines).strip()

        return extracted_text

    return None  # Jika tidak ditemukan Ingredients, kembalikan None

# 🔹 Streamlit UI
st.title("Halal and Haram Ingredients Checker 🍽️")
st.write("Select input method to check Halal or Haram status.")

option = st.radio("Select input method:", ["Manual Text Input", "Upload Image"])

# ======================= INPUT TEKS =======================
if option == "Manual Text Input":
    st.subheader("Input Ingredients")
    user_input = st.text_area("Enter text:", "")

    if st.button("Prediksi"):
        if user_input:
            st.write("**📄 Inputted text:**")
            st.write(user_input)

            label, confidence, warning_ingredients = predict_label(user_input)
            st.success(f"**Prediction: {label}**")
            st.info(f"**Confidence: {confidence}%**")

            if warning_ingredients:
                st.warning(f"⚠ This product contains ingredients that you should be aware of : {', '.join(warning_ingredients)}")

# ======================= UPLOAD GAMBAR =======================
elif option == "Upload Image":
    st.subheader("Upload Ingredients Label")
    uploaded_image = st.file_uploader("Upload Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        st.image(uploaded_image, use_container_width=True)

        with st.spinner("🔄 Analyzing image..."):
            extracted_text = extract_text_from_image(uploaded_image)

        if extracted_text:
            st.write("**📄 Extracted Text:**")
            st.write(extracted_text)

            label, confidence, warning_ingredients = predict_label(extracted_text)
            st.success(f"**Prediction: {label}**")
            st.info(f"**Confidence: {confidence}%**")

            if warning_ingredients:
                st.warning(f"⚠ This product contains ingredients that you should be aware of : {', '.join(warning_ingredients)}")
        else:
            st.error("⚠ Image does not contain ingredients. Please re-upload an image with ingredient labels.")
