import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import joblib

# --- HACK: Menangani isu 'numpy._core' agar database.pkl terbaca mulus ---
try:
    import numpy.core
    sys.modules['numpy._core'] = numpy.core
except ImportError:
    pass

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape,
    Conv2DTranspose, UpSampling2D,
)
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
IMG_SIZE        = (128, 128)
IMG_SHAPE       = (128, 128, 3)
LATENT_DIM      = 256
TOP_K           = 5
ENCODER_WEIGHTS = "encoder_only_weights.weights.h5"
DATABASE_PATH   = "database.pkl"
LOOKUP_CSV      = "images.csv"

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "ShirtFinder — Visual Search",
    page_icon  = "🔍",
    layout     = "wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: linear-gradient(135deg, #0A0A14 0%, #0D0D20 30%, #0F0A1A 60%, #0A0F18 100%);
    min-height: 100vh;
}

.hero-wrapper {
    position: relative;
    border-radius: 20px;
    padding: 52px 60px 44px;
    margin-bottom: 36px;
    background: linear-gradient(135deg, #13102A 0%, #0E1A2E 50%, #111228 100%);
    border: 1px solid rgba(120, 80, 255, 0.15);
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.4rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.0;
    margin: 0 0 10px;
}
.hero-title .accent {
    background: linear-gradient(90deg, #A855F7, #3B82F6, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #FFFFFF !important;
    margin-bottom: 12px;
}

.result-card {
    background: linear-gradient(160deg, rgba(20,16,42,0.98) 0%, rgba(12,18,36,0.98) 100%);
    border: 1px solid rgba(120, 80, 255, 0.2);
    border-radius: 16px;
    padding: 16px;
    height: 100%;
}

.score-value { font-weight: 700; color: #C084FC; }

/* Radio & Uploader Label Fix */
div[data-testid="stWidgetLabel"] label p {
    color: #FFFFFF !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODEL & LOOKUP DATA
# ─────────────────────────────────────────────────────────────
def build_autoencoder(input_shape=IMG_SHAPE, latent_dim=LATENT_DIM):
    inputs = Input(shape=input_shape, name="encoder_input")
    x = Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D((2,2), padding="same")(x)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x)
    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x)
    x = Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x)
    x = Flatten()(x)
    encoded = Dense(latent_dim, activation="relu", name="latent_space")(x)
    return Model(inputs, encoded, name="encoder")

@st.cache_resource
def load_encoder():
    encoder = build_autoencoder()
    encoder.load_weights(ENCODER_WEIGHTS)
    return encoder

@st.cache_data
def load_database():
    db = joblib.load(DATABASE_PATH)
    return db["embeddings"], db["metadata"].reset_index(drop=True)

@st.cache_data
def load_image_lookup():
    if os.path.exists(LOOKUP_CSV):
        df_img = pd.read_csv(LOOKUP_CSV)
        if 'id' in df_img.columns and 'link' in df_img.columns:
            return {str(row['id']): str(row['link']) for _, row in df_img.iterrows()}
    return {}

# ─────────────────────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-inner">
        <h1 class="hero-title">Shirt<span class="accent">Finder</span></h1>
        <p style="color:rgba(180,170,220,0.7)">Visual Search · Autoencoder + Lookup Table</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load Resources
if not os.path.exists(ENCODER_WEIGHTS) or not os.path.exists(DATABASE_PATH):
    st.error("Artefak model (weights/database) tidak ditemukan!")
    st.stop()

encoder_model = load_encoder()
db_embeddings, db_df = load_database()
image_lookup = load_image_lookup()

# Input Section
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown('<p class="section-label">📥 Input Gambar Query</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload kemeja", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file:
        query_img = Image.open(uploaded_file)
        st.image(query_img, caption="Preview Query", use_column_width=True)

with right_col:
    if uploaded_file:
        st.markdown('<p class="section-label">✨ Hasil Rekomendasi</p>', unsafe_allow_html=True)
        
        # Preprocess & Predict
        img_arr = img_to_array(query_img.convert("RGB").resize(IMG_SIZE)) / 255.0
        query_emb = encoder_model.predict(np.expand_dims(img_arr, axis=0), verbose=0)
        
        # Similarity Search
        scores = cosine_similarity(query_emb, db_embeddings)[0]
        indices = np.argsort(scores)[::-1][:TOP_K]
        
        cols = st.columns(TOP_K, gap="small")
        for rank, idx in enumerate(indices, start=1):
            row = db_df.iloc[idx]
            with cols[rank-1]:
                pid = str(row['id'])  # <-- paksa string
                img_url = image_lookup.get(pid)
                st.write(list(image_lookup.items())[:3])  # lihat sample key-value
                st.write(type(db_df['id'].iloc[0]))       # lihat tipe id di database
                st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f"""
                    <p style='color:#AAA; font-size:0.7rem; margin:10px 0 0;'>#{rank}</p>
                    <p style='font-size:0.85rem; color:#FFF;'>Match: <span class="score-value">{scores[idx]:.3f}</span></p>
                    <p style='color:rgba(140,120,180,0.5); font-size:0.7rem;'>ID: {pid}</p>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Silakan upload gambar untuk melihat hasil.")
