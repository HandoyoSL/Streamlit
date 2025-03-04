import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

def load_model():
    with open("best_model.pkl", "rb") as file:
        data = pickle.load(file)
    return data["model"], data["threshold"]

# Load model
model, threshold = load_model()

# Pastikan folder 'static' ada untuk menyimpan file CSV
if not os.path.exists("static"):
    os.makedirs("static")

# Streamlit UI
st.title("Prediksi dengan Model Machine Learning")

uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Pastikan urutan kolom sesuai dengan model
        features = [
            "age", "age_group", "job", "marital", "education", "housing", "loan",
            "contact", "month", "day_of_week", "campaign", "pdays", "previous",
            "poutcome", "anomali_cpp", "emp.var.rate", "cons.price.idx",
            "cons.conf.idx", "euribor3m", "nr.employed"
        ]
        
        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            st.error(f"Kolom berikut hilang di file CSV: {missing_cols}")
        else:
            df = df[features]
            
            # Prediksi probabilitas
            prob = model.predict_proba(df)[:, 1]
            
            # Tentukan hasil berdasarkan threshold
            df["prediction"] = np.where(prob >= threshold, "Yes", "No")
            
            # Simpan hasil prediksi
            csv_path = "static/predictions.csv"
            df.to_csv(csv_path, index=False)
            
            st.success("Prediksi berhasil dilakukan!")
            st.dataframe(df.head())
            
            # Tombol untuk mengunduh file hasil prediksi
            with open(csv_path, "rb") as file:
                st.download_button(
                    label="Unduh Hasil Prediksi",
                    data=file,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
