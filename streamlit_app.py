import streamlit as st
from src.sentiment_model import get_sentiment_details

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="😊",
    layout="centered",
)

st.title("😊 Análisis de Sentimientos en Reviews de Hoteles")

user_input = st.text_area(
    "¡Cuéntame tu experiencia! Escribe una review y descubre el sentimiento:",
    height=150,
)

if st.button("Analizar", type="primary"):
    if user_input.strip():
        details = get_sentiment_details(user_input)
        sentiment = details['sentiment']
        polarity = details['polarity']
        confidence = details['confidence']
        pos_words = details['positive_keywords']
        neg_words = details['negative_keywords']

        col1, col2, col3 = st.columns(3)
        col1.metric("Sentimiento", sentiment)
        col2.metric("Polaridad", f"{polarity:.2f}")
        col3.metric("Confianza", f"{confidence:.2%}")

        if pos_words:
            st.success(f"✨ Palabras positivas detectadas: {', '.join(pos_words)}")
        if neg_words:
            st.error(f"⚠️ Palabras negativas detectadas: {', '.join(neg_words)}")

        st.divider()
        st.caption("💡 Tip: El modelo funciona mejor en inglés. Para español, los resultados pueden variar.")
    else:
        st.warning("Por favor, escribe una review antes de analizar.")
