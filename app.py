import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

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

# --- 1. MODE KAMERA (NATIVE STREAMLIT) ---
if option == "Kamera Langsung (Snapshot)":
    st.header("Deteksi Wajah via Kamera")
    st.info("Klik tombol 'Take Photo' di bawah untuk mengambil gambar wajah Anda secara real-time.")
    
    # Input Kamera Bawaan Streamlit (Stabil di Cloud)
    img_file = st.camera_input("Ambil Foto")

    if img_file is not None:
        # Konversi file upload ke Image
        image = Image.open(img_file)
        img_array = np.array(image)
        
        # Prediksi YOLO
        results = model.predict(img_array, conf=conf)
        res_plot = results[0].plot()
        
        # Tampilkan Hasil
        st.image(res_plot, caption="Hasil Deteksi Saat Ini", use_container_width=True)
        
        # Hitung Statistik Sederhana
        boxes = results[0].boxes
        if len(boxes) > 0:
            names = model.names
            classes = boxes.cls.cpu().numpy()
            detected = [names[int(c)] for c in classes]
            
            if "no_mask" in detected or "incorrect_mask" in detected:
                st.error("⚠️ PELANGGARAN TERDETEKSI! Mohon gunakan masker dengan benar.")
            else:
                st.success("✅ PROTOKOL DIPATUHI. Terima kasih.")
                
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


# --- 3. MODE UPLOAD VIDEO (FIXED FOR LINUX SERVER) ---
elif option == "Upload Video":
    st.header("Deteksi pada Video")
    uploaded_video = st.file_uploader("Upload file video (MP4)", type=['mp4'])
    
    if uploaded_video is not None:
        # Simpan file input sementara
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        st.warning("Sedang memproses video... Harap tunggu. Jangan refresh halaman.")
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Ambil properti video input
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
            
            # --- SOLUSI UTAMA: GANTI KE WEBM (VP9) ---
            # VP9 (vp09) adalah codec open source yang PASTI ada di Linux server
            # Format file output harus .webm agar bisa diputar di browser
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
            
            # Coba codec VP9 (kualitas tinggi, open source)
            fourcc = cv2.VideoWriter_fourcc(*'vp09') 
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                 # Fallback ke VP8 jika VP9 gagal (sangat jarang terjadi)
                 st.warning("Codec VP9 gagal, mencoba VP8...")
                 fourcc = cv2.VideoWriter_fourcc(*'vp80')
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
            if os.path.exists(output_path):
                # Buka file output dalam mode binary read ('rb')
                with open(output_path, 'rb') as f:
                    video_bytes = f.read()
                st.video(video_bytes, format="video/webm") # Spesifik format webm
            else:
                st.error("File output tidak ditemukan.")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses video: {e}")
        
        finally:
            # Bersihkan file temporary
            if os.path.exists(video_path):
                os.unlink(video_path)
            # File output dibiarkan agar st.video bisa memutarnya, 
            # Streamlit akan membersihkannya nanti saat sesi berakhir


