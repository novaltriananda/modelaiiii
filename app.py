import streamlit as st
import pandas as pd
import pickle

# Fungsi untuk load model dari file .pkl
@st.cache_resource
def load_model_pickle():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Fungsi untuk load scaler dari file .pkl
@st.cache_resource
def load_scaler():
    with open("scaler2.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

# Load model dan scaler
model = load_model_pickle()
scaler = load_scaler()

# Judul aplikasi
st.title("Prediksi Model Machine Learning dengan Data Excel")

# Upload file Excel
uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx", "xls"])

if uploaded_file:
    try:
        # Membaca file Excel
        data = pd.read_excel(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(data)

        # Validasi: Pastikan hanya kolom numerik yang di-scaling
        try:
            numeric_data = data.select_dtypes(include=["number"])
            data_scaled = scaler.transform(numeric_data)
        except Exception as e:
            st.error("Error saat scaling data. Pastikan data hanya berisi kolom numerik.")
            st.stop()

        # Prediksi menggunakan model
        if st.button("Prediksi"):
            predictions = model.predict(data_scaled)  # Menggunakan data yang sudah di-scaling
            data["Prediction"] = predictions  # Jika output adalah hasil prediksi langsung

            st.success("Prediksi berhasil!")
            st.write("Hasil Prediksi:")
            st.write(data)

            # Tombol unduh hasil prediksi
            st.download_button(
                label="Unduh Hasil Prediksi",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Error: {e}")
