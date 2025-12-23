import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os
import subprocess  # Library untuk menjalankan perintah terminal (FFmpeg)

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Deteksi Masker Wajah",
    page_icon="😷",
    layout="wide"
)

# --- KONFIGURASI WEBRTC ---
# Menambahkan server STUN Google agar tembus firewall Cloud
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
    st.error(f"Model error: {e}")
    st.stop()

# --- SIDEBAR ---
option = st.sidebar.selectbox(
    "Pilih Mode Input",
    ("Webcam Real-time", "Upload Gambar", "Upload Video")
)
conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# --- 1. MODE WEBCAM ---
if option == "Webcam Real-time":
    st.header("Deteksi Real-time via Webcam")
    st.info("Jika error, refresh halaman. Tunggu 10-20 detik untuk inisialisasi kamera.")
    
    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Prediksi
            results = model(img, conf=conf)
            annotated_frame = results[0].plot()
            
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_streamer(
        key="mask-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# --- 2. MODE GAMBAR ---
elif option == "Upload Gambar":
    st.header("Deteksi pada Gambar")
    uploaded_file = st.file_uploader("Upload file gambar", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        
        results = model.predict(img_array, conf=conf)
        res_plot = results[0].plot()
        with col2:
            st.image(res_plot, caption="Hasil Deteksi", use_container_width=True)

# --- 3. MODE VIDEO (FFMPEG WRAPPER) ---
elif option == "Upload Video":
    st.header("Deteksi pada Video")
    uploaded_video = st.file_uploader("Upload file video (MP4)", type=['mp4'])
    
    if uploaded_video is not None:
        # Simpan input
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        input_path = tfile.name
        
        st.warning("Sedang memproses... (Mungkin butuh waktu tergantung durasi video)")
        
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Output SEMENTARA sebagai .avi dengan codec MJPG (OpenCV pasti support ini)
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        # Proses Frame-by-Frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.predict(frame, conf=conf, verbose=False)
            res_frame = results[0].plot()
            out.write(res_frame)
            
            frame_count += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
            
        cap.release()
        out.release()
        
        # --- TAHAP KONVERSI (Magic Step) ---
        # Kita gunakan FFmpeg (terminal) untuk ubah AVI -> MP4 (H.264) agar bisa diplay browser
        st.info("Mengonversi format video agar kompatibel dengan browser...")
        final_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # Perintah terminal untuk konversi
        subprocess.call(args=[
            'ffmpeg', '-y', '-i', temp_output, '-c:v', 'libx264', final_output
        ])
        
        st.success("Selesai!")
        st.video(final_output)
        
        # Bersihkan sampah
        try:
            os.unlink(input_path)
            os.unlink(temp_output)
            # os.unlink(final_output) # Jangan hapus final output agar bisa ditonton
        except:
            pass
