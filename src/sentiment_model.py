from textblob import TextBlob
import re
import pandas as pd

# Palabras clave que suelen aparecer en reseñas de hoteles. ¡Ayudan a afinar el análisis!
HOTEL_POSITIVE_KEYWORDS = {
    'amazing', 'excellent', 'perfect', 'beautiful', 'wonderful', 'great', 'clean',
    'comfortable', 'friendly', 'helpful', 'spacious', 'modern', 'luxury', 'stunning',
    'outstanding', 'fantastic', 'impressive', 'delicious', 'convenient', 'recommended'
}

HOTEL_NEGATIVE_KEYWORDS = {
    'terrible', 'awful', 'dirty', 'noisy', 'small', 'outdated', 'poor', 'rude',
    'unhelpful', 'disappointing', 'overpriced', 'basic', 'limited', 'cold', 'old',
    'broken', 'uncomfortable', 'crowded', 'slow', 'expensive'
}

def preprocess_text(text):
    """
    Prepara el texto para el análisis de sentimientos.
    Aquí limpiamos y normalizamos para que el modelo entienda mejor el mensaje real.
    """
    # Convertir a minúsculas para evitar confusiones
    text = text.lower()
    # Eliminar caracteres raros pero dejar puntuación útil
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    # Quitar espacios extra
    text = ' '.join(text.split())
    return text

def calculate_domain_sentiment(text):
    """
    Calcula un puntaje de sentimiento usando palabras clave típicas del sector hotelero.
    Así, el análisis es más relevante para este dominio.
    """
    words = set(text.lower().split())
    positive_score = len(words.intersection(HOTEL_POSITIVE_KEYWORDS))
    negative_score = len(words.intersection(HOTEL_NEGATIVE_KEYWORDS))
    if positive_score == 0 and negative_score == 0:
        return 0
    return (positive_score - negative_score) / (positive_score + negative_score)

def analyze_sentiment(text):
    """
    Analiza el sentimiento de un texto combinando la magia de TextBlob con el toque especial del sector hotelero.
    Devuelve la polaridad y una etiqueta amigable (Positivo, Negativo o Neutral).
    Args:
        text (str): El texto a analizar
    Returns:
        tuple: (polaridad, sentimiento)
            - polaridad: float entre -1 y 1
            - sentimiento: str ('Positivo', 'Negativo', o 'Neutral')
    """
    # Preparamos el texto para que el análisis sea más preciso
    processed_text = preprocess_text(text)
    # TextBlob hace su magia aquí
    blob = TextBlob(processed_text)
    textblob_polarity = blob.sentiment.polarity
    # Añadimos el toque hotelero
    domain_polarity = calculate_domain_sentiment(processed_text)
    # Combinamos ambos resultados para un veredicto más justo
    combined_polarity = (textblob_polarity * 0.6) + (domain_polarity * 0.4)
    # Definimos los umbrales de cada sentimiento
    if combined_polarity > 0.15:
        sentiment = 'Positivo'
    elif combined_polarity < -0.15:
        sentiment = 'Negativo'
    else:
        sentiment = 'Neutral'
    return combined_polarity, sentiment

def batch_analyze(df, text_column='review'):
    """
    Analiza una columna completa de reseñas en un DataFrame de pandas.
    Ideal para cuando tienes muchas opiniones y poco tiempo.
    Args:
        df (pandas.DataFrame): DataFrame con la columna de texto a analizar
        text_column (str): Nombre de la columna que contiene los textos
    Returns:
        pandas.DataFrame: DataFrame original con columnas extra de polaridad y sentimiento
    """
    # Trabajamos sobre una copia para no tocar tus datos originales
    result_df = df.copy()
    # Analizamos cada reseña
    sentiments = result_df[text_column].apply(analyze_sentiment)
    # Separamos los resultados en columnas
    result_df['polarity'], result_df['sentiment'] = zip(*sentiments)
    return result_df

def get_sentiment_details(text):
    """
    Devuelve un análisis detallado y transparente del sentimiento de un texto.
    Incluye palabras clave encontradas y un nivel de confianza.
    Args:
        text (str): El texto a analizar
    Returns:
        dict: Diccionario con detalles del análisis
    """
    polarity, sentiment = analyze_sentiment(text)
    # Buscamos palabras clave positivas y negativas
    words = set(preprocess_text(text).split())
    positive_words = words.intersection(HOTEL_POSITIVE_KEYWORDS)
    negative_words = words.intersection(HOTEL_NEGATIVE_KEYWORDS)
    return {
        'polarity': polarity,
        'sentiment': sentiment,
        'positive_keywords': list(positive_words),
        'negative_keywords': list(negative_words),
        'confidence': abs(polarity)  # La confianza es la magnitud de la polaridad
    }
