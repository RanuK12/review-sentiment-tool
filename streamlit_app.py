import streamlit as st
from src.sentiment_model import analyze_sentiment

# Título amigable para la app
st.title("😊 Análisis de Sentimientos en Reviews de Hoteles")

# Instrucción cálida para el usuario
user_input = st.text_area("¡Cuéntame tu experiencia! Escribe una review y descubre el sentimiento:")

if st.button("Analizar"):
    if user_input.strip():
        polarity, sentiment = analyze_sentiment(user_input)
        st.success(f"✨ <b>Sentimiento:</b> {sentiment}", icon="💡")
        st.info(f"<b>Polaridad:</b> {polarity:.2f}", icon="📊")
    else:
        st.warning("Por favor, escribe una review antes de analizar.")
