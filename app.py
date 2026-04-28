# app.py
"""
Streamlit App — Visually Similar Product Recommendation
Dataset  : Fashion Product Images (Shirts only)
Model    : Convolutional Autoencoder (encoder_weights.h5)
Database : Embeddings dari test split (database.pkl)

Run : streamlit run app.py
"""

import os
import numpy as np
import streamlit as st
from PIL import Image
import joblib

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

/* ── Global ───────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── App background: deep dark ombre ─────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0A0A14 0%, #0D0D20 30%, #0F0A1A 60%, #0A0F18 100%);
    min-height: 100vh;
}

/* ── Hero ─────────────────────────────────────────────────── */
.hero-wrapper {
    position: relative;
    border-radius: 20px;
    padding: 52px 60px 44px;
    margin-bottom: 36px;
    overflow: hidden;
    background: linear-gradient(135deg, #13102A 0%, #0E1A2E 50%, #111228 100%);
    border: 1px solid rgba(120, 80, 255, 0.15);
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -80px; left: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(120, 60, 255, 0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-wrapper::after {
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(0, 180, 255, 0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-inner {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 24px;
    flex-wrap: wrap;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.4rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.0;
    margin: 0 0 10px;
    letter-spacing: -0.02em;
}
.hero-title .accent {
    background: linear-gradient(90deg, #A855F7, #3B82F6, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-size: 0.9rem;
    color: rgba(180, 170, 220, 0.7);
    margin: 0;
    font-weight: 300;
    letter-spacing: 0.03em;
}
.hero-badge {
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid rgba(168, 85, 247, 0.35);
    color: #C084FC;
    font-size: 0.76rem;
    font-weight: 600;
    padding: 8px 18px;
    border-radius: 100px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    white-space: nowrap;
    backdrop-filter: blur(6px);
}

/* ── Section label ───────────────────────────────────────── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(150, 130, 200, 0.7);
    margin-bottom: 12px;
}

/* ── Stat chips ──────────────────────────────────────────── */
.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(120, 80, 255, 0.2);
    border-radius: 100px;
    padding: 7px 16px;
    font-size: 0.82rem;
    color: rgba(200, 185, 240, 0.85);
    font-weight: 400;
    backdrop-filter: blur(4px);
}
.stat-chip strong {
    color: #E2D9FF;
    font-weight: 600;
}

/* ── Divider ─────────────────────────────────────────────── */
.custom-divider {
    border: none;
    border-top: 1px solid rgba(120, 80, 255, 0.12);
    margin: 28px 0;
}

/* ── Upload & input panel ─────────────────────────────────── */
.upload-panel {
    background: linear-gradient(160deg, rgba(20,16,40,0.95) 0%, rgba(14,18,38,0.95) 100%);
    border: 1px solid rgba(120, 80, 255, 0.18);
    border-radius: 18px;
    padding: 28px;
}

/* ── Query preview card ──────────────────────────────────── */
.query-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(120, 80, 255, 0.18);
    border-radius: 14px;
    padding: 18px;
    margin-top: 16px;
}
.query-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: rgba(150, 130, 200, 0.7);
    margin-bottom: 10px;
}

/* ── Result heading ──────────────────────────────────────── */
.results-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #EDE9FF;
    margin: 6px 0 22px;
    letter-spacing: -0.01em;
}

/* ── Result card ─────────────────────────────────────────── */
.result-card {
    background: linear-gradient(160deg, rgba(20,16,42,0.98) 0%, rgba(12,18,36,0.98) 100%);
    border: 1px solid rgba(120, 80, 255, 0.2);
    border-radius: 16px;
    padding: 16px;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(168,85,247,0.5), transparent);
}
.result-card:hover {
    transform: translateY(-4px);
    border-color: rgba(168, 85, 247, 0.45);
    box-shadow: 0 12px 40px rgba(120, 60, 255, 0.18);
}

.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7C3AED, #2563EB);
    color: #FFFFFF;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 11px;
    border-radius: 100px;
    margin-bottom: 10px;
    letter-spacing: 0.06em;
}
.rank-badge-1 { background: linear-gradient(135deg, #A855F7, #EC4899); }
.rank-badge-2 { background: linear-gradient(135deg, #7C3AED, #3B82F6); }
.rank-badge-3 { background: linear-gradient(135deg, #2563EB, #06B6D4); }
.rank-badge-4 { background: linear-gradient(135deg, #0891B2, #10B981); }
.rank-badge-5 { background: linear-gradient(135deg, #059669, #84CC16); }

.score-text {
    font-size: 0.78rem;
    color: rgba(180, 165, 220, 0.7);
    margin-top: 10px;
}
.score-value {
    font-weight: 700;
    color: #C084FC;
}
.product-id {
    font-size: 0.72rem;
    color: rgba(140, 120, 180, 0.5);
    margin-top: 3px;
}

/* ── Score bar ───────────────────────────────────────────── */
.score-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 3px;
    margin-top: 8px;
    overflow: hidden;
}
.score-bar-fill {
    background: linear-gradient(90deg, #7C3AED, #06B6D4);
    height: 3px;
    border-radius: 100px;
}

/* ── Empty state ─────────────────────────────────────────── */
.empty-state {
    height: 360px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: linear-gradient(160deg, rgba(20,16,42,0.6) 0%, rgba(12,18,36,0.6) 100%);
    border: 1.5px dashed rgba(120, 80, 255, 0.22);
    border-radius: 18px;
    color: rgba(160, 140, 210, 0.5);
    gap: 14px;
}
.empty-icon {
    font-size: 2.8rem;
    opacity: 0.5;
}
.empty-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.88rem;
    margin: 0;
    font-weight: 300;
    letter-spacing: 0.01em;
}

/* ── Error box ───────────────────────────────────────────── */
.error-box {
    background: rgba(220, 38, 38, 0.08);
    border: 1px solid rgba(220, 38, 38, 0.3);
    border-radius: 14px;
    padding: 20px 26px;
    color: #FCA5A5;
    font-size: 0.9rem;
}
.error-box code {
    background: rgba(220, 38, 38, 0.15);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85rem;
}

/* ── Footer ──────────────────────────────────────────────── */
.footer {
    text-align: center;
    font-size: 0.76rem;
    color: rgba(130, 115, 175, 0.45);
    font-family: 'Space Grotesk', sans-serif;
    padding: 8px 0 20px;
}

/* ── Streamlit radio overrides ───────────────────────────── */
div[data-testid="stRadio"] label {
    font-size: 0.88rem !important;
    color: rgba(200, 185, 240, 0.85) !important;
}
div[data-testid="stRadio"] > div {
    gap: 10px !important;
}

/* ── File uploader overrides ─────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(120, 80, 255, 0.3) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.02) !important;
    padding: 12px !important;
}
[data-testid="stFileUploader"] label {
    color: rgba(190, 175, 235, 0.85) !important;
    font-size: 0.88rem !important;
}

/* ── Camera input overrides ──────────────────────────────── */
[data-testid="stCameraInput"] {
    border-radius: 12px !important;
}

/* ── Spinner text ────────────────────────────────────────── */
.stSpinner > div {
    color: #A855F7 !important;
}

/* ── Warning override ────────────────────────────────────── */
[data-testid="stWarning"] {
    background: rgba(245, 158, 11, 0.08) !important;
    border-color: rgba(245, 158, 11, 0.25) !important;
    color: #FCD34D !important;
}

/* ── Column gap override ─────────────────────────────────── */
[data-testid="column"] {
    padding: 0 6px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────
def build_autoencoder(input_shape=IMG_SHAPE, latent_dim=LATENT_DIM):
    inputs = Input(shape=input_shape, name="encoder_input")

    x = Conv2D(32,  (3, 3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(64,  (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    conv_shape = x.shape[1:]
    x       = Flatten()(x)
    encoded = Dense(latent_dim, activation="relu", name="latent_space")(x)

    x = Dense(conv_shape[0] * conv_shape[1] * conv_shape[2], activation="relu")(encoded)
    x = Reshape(conv_shape)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(256, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(128, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(64,  (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2DTranspose(3, (3, 3), activation="sigmoid", padding="same",
                              name="decoder_output")(x)

    autoencoder = Model(inputs, decoded, name="autoencoder")
    encoder     = Model(inputs, encoded, name="encoder")
    return autoencoder, encoder


# ─────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat encoder model…")
def load_encoder():
    autoencoder, encoder = build_autoencoder()
    encoder.load_weights(ENCODER_WEIGHTS)
    return encoder


@st.cache_data(show_spinner="Memuat embedding database…")
def load_database():
    db         = joblib.load(DATABASE_PATH)
    embeddings = db["embeddings"]
    metadata   = db["metadata"].reset_index(drop=True)
    return embeddings, metadata


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────
def preprocess_image(pil_image):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def find_similar(pil_image, encoder_model, db_embeddings, db_df):
    query_emb = encoder_model.predict(preprocess_image(pil_image), verbose=0)
    scores    = cosine_similarity(query_emb, db_embeddings)[0]
    indices   = np.argsort(scores)[::-1][:TOP_K]
    results   = db_df.iloc[indices].copy().reset_index(drop=True)
    results["similarity_score"] = scores[indices]
    return results


# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-inner">
        <div>
            <h1 class="hero-title">Shirt<span class="accent">Finder</span></h1>
            <p class="hero-subtitle">
                Visual search · Convolutional Autoencoder + Cosine Similarity
            </p>
        </div>
        <div>
            <span class="hero-badge">🧵 Shirts Only · Top-5 Results</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CEK ARTEFAK
# ─────────────────────────────────────────────────────────────
missing = [f for f in [ENCODER_WEIGHTS, DATABASE_PATH] if not os.path.exists(f)]
if missing:
    st.markdown(f"""
    <div class="error-box">
        <strong>⚠️ File tidak ditemukan:</strong> {', '.join(f'<code>{f}</code>' for f in missing)}<br><br>
        Jalankan <code>autoencoder_training_revised.ipynb</code> terlebih dahulu untuk
        men-generate <code>encoder_weights.h5</code> dan <code>database.pkl</code>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

encoder_model        = load_encoder()
db_embeddings, db_df = load_database()


# ─────────────────────────────────────────────────────────────
# STATS CHIPS
# ─────────────────────────────────────────────────────────────
col_s1, col_s2, col_s3, _ = st.columns([1, 1, 1, 4])
with col_s1:
    st.markdown(f'<div class="stat-chip">🗂 Database&nbsp;<strong>{len(db_df):,} items</strong></div>',
                unsafe_allow_html=True)
with col_s2:
    st.markdown(f'<div class="stat-chip">🧠 Latent&nbsp;<strong>{LATENT_DIM}d</strong></div>',
                unsafe_allow_html=True)
with col_s3:
    st.markdown(f'<div class="stat-chip">🔎 Top-K&nbsp;<strong>{TOP_K}</strong></div>',
                unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown('<p class="section-label">📥 Input Gambar Query</p>', unsafe_allow_html=True)

    input_method = st.radio(
        "Pilih metode input:",
        options=["📁 Upload dari galeri", "📷 Foto dengan kamera"],
        horizontal=False,
        label_visibility="collapsed",
    )

    query_image = None

    if input_method == "📁 Upload dari galeri":
        uploaded_file = st.file_uploader(
            "Pilih gambar produk (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )
        if uploaded_file:
            query_image = Image.open(uploaded_file)

    else:
        camera_photo = st.camera_input("Arahkan kamera ke produk")
        if camera_photo:
            query_image = Image.open(camera_photo)

    if query_image:
        st.markdown('<div class="query-card">', unsafe_allow_html=True)
        st.markdown('<p class="query-label">Preview Query</p>', unsafe_allow_html=True)
        st.image(query_image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# RESULT SECTION
# ─────────────────────────────────────────────────────────────
RANK_BADGE_CLASSES = ["rank-badge-1", "rank-badge-2", "rank-badge-3", "rank-badge-4", "rank-badge-5"]

with right_col:
    if query_image:
        st.markdown('<p class="section-label">✨ Hasil Rekomendasi</p>', unsafe_allow_html=True)

        with st.spinner("Mencari produk serupa…"):
            results = find_similar(query_image, encoder_model, db_embeddings, db_df)

        st.markdown(
            f'<p class="results-heading">Top {TOP_K} Produk Paling Mirip</p>',
            unsafe_allow_html=True,
        )

        cols = st.columns(TOP_K, gap="small")
        for rank, (col, (_, row)) in enumerate(zip(cols, results.iterrows()), start=1):
            with col:
                score_pct   = int(row["similarity_score"] * 100)
                badge_class = RANK_BADGE_CLASSES[rank - 1]

                st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                st.markdown(
                    f'<span class="rank-badge {badge_class}">#{rank}</span>',
                    unsafe_allow_html=True
                )

                # ── SOLUSI: PAKAI URL DARI ID, JANGAN PAKAI FILENAME KAGGLE ──
                # product_id = str(row["id"])
                product_id = str(int(row["id"])) # Pastikan ID bersih dari .0
                img_url = f"https://assets.myntassets.com/assets/images/{product_id}.jpg"

                # Gunakan parameter width=None agar mengikuti container CSS
                # Tambahkan caption agar jika gambar error, kita tahu ID mana yang bermasalah
                try:
                    st.image(img_url, use_column_width=True)
                except:
                    st.warning(f"Gagal memuat gambar ID: {product_id}")
                    
                # Langsung tampilkan pake URL
                # st.image(img_url, use_column_width=True)

                st.markdown(f"""
                    <p class="score-text">
                        Similarity&nbsp;
                        <span class="score-value">{row['similarity_score']:.4f}</span>
                    </p>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{score_pct}%"></div>
                    </div>
                    <p class="product-id">ID: {product_id}</p>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown("""
<p class="footer">
    ShirtFinder · Convolutional Autoencoder + Cosine Similarity ·
    Dataset: <em>paramaggarwal/fashion-product-images-dataset</em>
</p>
""", unsafe_allow_html=True)
