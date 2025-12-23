import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Deteksi Masker Wajah",
    page_icon="😷",
    layout="wide"
)

# --- KONFIGURASI WEBRTC (PENTING UNTUK CLOUD) ---
# Menggunakan Google STUN Server agar bisa tembus firewall
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.title("😷 Sistem Deteksi Kepatuhan Masker Wajah")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model tidak ditemukan. Error: {e}")
    st.stop()

# --- SIDEBAR MENU ---
option = st.sidebar.selectbox(
    "Pilih Mode Input",
    ("Webcam Real-time", "Upload Gambar", "Upload Video")
)

conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# --- 1. MODE WEBCAM (WEB COMPATIBLE) ---
if option == "Webcam Real-time":
    st.header("Deteksi Real-time via Webcam")
    st.info("Jika webcam tidak muncul, pastikan browser mengizinkan akses kamera dan tunggu hingga 10-20 detik.")
    
    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Prediksi YOLO
            results = model(img, conf=conf)
            annotated_frame = results[0].plot()
            
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_streamer(
        key="mask-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION, # <--- INI KUNCINYA
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
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        
        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)
        
        results = model.predict(img_array, conf=conf)
        res_plot = results[0].plot()
        
        with col2:
            st.image(res_plot, caption="Hasil Deteksi", use_container_width=True)

# --- 3. MODE UPLOAD VIDEO (FIXED CODEC) ---
elif option == "Upload Video":
    st.header("Deteksi pada Video")
    uploaded_video = st.file_uploader("Upload file video (MP4)", type=['mp4'])
    
    if uploaded_video is not None:
        # Simpan file input sementara
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        st.warning("Sedang memproses video... Harap tunggu. Jangan refresh halaman.")
        
        cap = cv2.VideoCapture(video_path)
        
        # Persiapan output video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # --- PERBAIKAN CODEC DISINI ---
        # Menggunakan 'avc1' (H.264) agar bisa diputar di browser (Chrome/Edge/HP)
        # Jika avc1 gagal, dia akan fallback ke mp4v (tapi mp4v sering blank di web)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # Coba codec H.264
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Prediksi
            results = model.predict(frame, conf=conf, verbose=False)
            res_frame = results[0].plot()
            
            # Tulis ke file output
            out.write(res_frame)
            
            # Update progress bar
            frame_count += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
            
        cap.release()
        out.release()
        
        st.success("Pemrosesan Selesai!")
        
        # Tampilkan Video Hasil
        # Buka file output dalam mode binary read ('rb')
        try:
            with open(output_path, 'rb') as f:
                video_bytes = f.read()
            st.video(video_bytes)
        except Exception as e:
            st.error(f"Gagal memuat video hasil: {e}")
        
        # Bersihkan file temporary
        os.unlink(video_path)
        # os.unlink(output_path) # Biarkan file output agar tidak error saat replay

