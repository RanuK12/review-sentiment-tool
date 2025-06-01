import streamlit as st
from src.sentiment_model import analyze_sentiment

st.title("Análisis de Sentimientos en Reviews")
user_input = st.text_area("Escribe una review:")

if st.button("Analizar"):
    polarity, sentiment = analyze_sentiment(user_input)
    st.write(f"**Sentimiento:** {sentiment}")
    st.write(f"**Polaridad:** {polarity:.2f}")
