import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ============================================================
# KONFIGURASI
# ============================================================
# Path dibuat relatif terhadap lokasi file app.py ini sendiri,
# supaya tetap ketemu model-nya walau Streamlit Cloud menjalankan
# app dari root repo (bukan dari folder apple-orange/v2/).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
H5_MODEL_PATH = os.path.join(BASE_DIR, "model_pretrained_mobilenetv2.h5")
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "model_appleorange_quant.tflite")
IMG_SIZE = (128, 128)
CLASS_NAMES = ["Apple", "Orange"]  # index 0 -> Apple, index 1 -> Orange

st.set_page_config(page_title="Apple vs Orange Classifier v2", page_icon="🍎", layout="centered")

st.markdown("""
<style>
h1 { color: #2F4B3C; }
.stButton>button { background: #2F4B3C; color: #FFFFFF; border: none; }
.stButton>button:hover { background: #22362A; color: #FFFFFF; }
div[role="radiogroup"] label { border: 1px solid #DCE3D0; border-radius: 4px; padding: 0.2rem 0.6rem; margin-right: 0.4rem; }
.stProgress > div > div { background: #2F4B3C; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model(H5_MODEL_PATH)


@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def predict_tflite(interpreter, processed):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], processed.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0][0]


# ============================================================
# PREPROCESSING
# ============================================================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)


# ============================================================
# SESSION STATE UNTUK RIWAYAT PREDIKSI
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# UI
# ============================================================
st.title("Apple vs Orange Classifier")
st.caption("Versi 2")
st.write("Upload gambar apel atau jeruk untuk diprediksi. Versi ini menambahkan breakdown "
         "probabilitas kedua kelas, riwayat prediksi, dan opsi model teroptimasi (TFLite).")

model_choice = st.radio(
    "Pilih model inferensi",
    ["Model asli (.h5)", "Model teroptimasi (.tflite - quantized)"],
    horizontal=True,
)

uploaded_file = st.file_uploader(
    "Pilih gambar...", type=["jpg", "jpeg", "png"],
    help="Gunakan foto satu buah dengan latar belakang polos untuk hasil terbaik.",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if st.button("Prediksi"):
        with st.spinner("Model sedang memproses..."):
            processed = preprocess_image(image)

            if model_choice.startswith("Model asli"):
                model = load_keras_model()
                pred = model.predict(processed)[0][0]
            else:
                interpreter = load_tflite_model()
                pred = predict_tflite(interpreter, processed)

            prob_orange = float(pred)
            prob_apple = 1 - prob_orange
            pred_class = CLASS_NAMES[1] if prob_orange > 0.5 else CLASS_NAMES[0]
            confidence = max(prob_apple, prob_orange)

        st.success(f"Prediksi: **{pred_class}** ({confidence * 100:.2f}% confidence)")

        st.write("Breakdown probabilitas:")
        st.write(f"🍎 Apple: {prob_apple * 100:.2f}%")
        st.progress(prob_apple)
        st.write(f"🍊 Orange: {prob_orange * 100:.2f}%")
        st.progress(prob_orange)

        st.session_state.history.append({
            "file": uploaded_file.name,
            "prediksi": pred_class,
            "confidence": f"{confidence * 100:.2f}%",
            "model": model_choice,
        })
else:
    st.info("Silakan upload gambar terlebih dahulu.")

if st.session_state.history:
    st.markdown("---")
    st.subheader("Riwayat Prediksi (sesi ini)")
    st.table(st.session_state.history)

st.markdown("---")
st.caption("Model transfer learning MobileNetV2, dengan opsi versi teroptimasi (TFLite).")
st.caption("Tugas Big Data LAS, iterasi kedua (Minggu 3–4).")
