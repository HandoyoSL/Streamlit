import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model dari file pickle
with open("best_model.pkl", "rb") as file:
    data = pickle.load(file)

model = data["model"]
threshold = 0.45  # Tetapkan threshold ke 0.45 (tetap, tapi tidak dipakai)

# Daftar fitur yang dibutuhkan model
features = [
    "age", "age_group", "job", "marital", "education", "housing", "loan",
    "contact", "month", "day_of_week", "campaign", "pdays", "previous",
    "poutcome", "anomali_cpp", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed"
]

# UI Streamlit
st.title("Prediksi dengan Model Terbaik")
st.write("Unggah file CSV untuk melakukan prediksi.")

# Upload file CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        # Baca CSV
        df = pd.read_csv(uploaded_file)
        st.write("Data yang diunggah:")
        st.dataframe(df.head())

        # Periksa apakah semua kolom yang dibutuhkan ada di file CSV
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            st.error(f"Kolom berikut hilang di file CSV: {missing_cols}")
        else:
            # Gunakan hanya kolom yang diperlukan
            df = df[features]

            # Prediksi probabilitas
            prob = model.predict_proba(df)[:, 1]  # Ambil probabilitas untuk kelas positif

            # Tampilkan probabilitas prediksi
            df["Prediksi Probabilitas"] = prob

            # Tampilkan hasil prediksi
            st.write("Hasil Prediksi (Probabilitas):")
            st.dataframe(df)

            # Buat file CSV untuk diunduh
            csv_output = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Hasil Prediksi",
                data=csv_output,
                file_name="hasil_prediksi.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
