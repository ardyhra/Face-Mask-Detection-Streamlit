import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Deteksi Masker Wajah",
    page_icon="😷",
    layout="wide"
)

st.title("😷 Sistem Deteksi Kepatuhan Masker Wajah")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Pastikan file best.pt sudah ada satu folder dengan app.py
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model tidak ditemukan. Pastikan file 'best.pt' ada di folder yang sama. Error: {e}")
    st.stop()

# --- SIDEBAR MENU ---
option = st.sidebar.selectbox(
    "Pilih Mode Input",
    ("Webcam Real-time", "Upload Gambar", "Upload Video")
)

conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# --- FUNGSI PROSES GAMBAR ---
def process_frame(frame, conf_threshold):
    results = model.predict(frame, conf=conf_threshold)
    return results[0].plot()

# --- 1. MODE WEBCAM (WEB COMPATIBLE) ---
if option == "Webcam Real-time":
    st.header("Deteksi Real-time via Webcam")
    st.write("Tunggu sebentar hingga kotak video muncul. Izinkan akses kamera browser.")
    
    # Class callback untuk memproses video stream
    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Prediksi
            results = model(img, conf=conf)
            annotated_frame = results[0].plot()
            
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_streamer(
        key="mask-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# --- 2. MODE UPLOAD GAMBAR ---
elif option == "Upload Gambar":
    st.header("Deteksi pada Gambar")
    uploaded_file = st.file_uploader("Upload file gambar (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # Baca gambar
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)
        
        # Prediksi
        res_image = process_frame(img_array, conf)
        
        with col2:
            st.image(res_image, caption="Hasil Deteksi", use_container_width=True)

# --- 3. MODE UPLOAD VIDEO ---
elif option == "Upload Video":
    st.header("Deteksi pada Video")
    uploaded_video = st.file_uploader("Upload file video (MP4)", type=['mp4'])
    
    if uploaded_video is not None:
        # Simpan video ke file sementara
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        vf = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()
        
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            
            # Proses frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res_frame = process_frame(frame, conf)
            
            # Tampilkan
            stframe.image(res_frame, caption="Sedang Memproses Video...", use_container_width=True)
            
        vf.release()