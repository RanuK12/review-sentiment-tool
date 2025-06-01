from textblob import TextBlob
import re
import pandas as pd

# Palabras clave que suelen aparecer en reseñas de hoteles. ¡Ayudan a afinar el análisis!
HOTEL_POSITIVE_KEYWORDS = {
    'amazing', 'excellent', 'perfect', 'beautiful', 'wonderful', 'great',
    'comfortable', 'friendly', 'helpful', 'spacious', 'modern', 'luxury',
    'stunning', 'outstanding', 'fantastic', 'impressive', 'delicious',
    'convenient', 'recommended'
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
    Analiza el sentimiento de un texto combinando TextBlob y palabras clave específicas del dominio.
    """
    # Preprocesar el texto
    processed_text = preprocess_text(text)
    
    # Calcular sentimiento con TextBlob
    blob = TextBlob(processed_text)
    textblob_polarity = blob.sentiment.polarity
    
    # Calcular sentimiento específico del dominio
    domain_polarity = calculate_domain_sentiment(processed_text)
    
    # Combinar ambos resultados (90% TextBlob, 10% palabras clave)
    combined_polarity = 0.9 * textblob_polarity + 0.1 * domain_polarity
    
    # Determinar el sentimiento basado en la polaridad combinada
    if combined_polarity > 0.3:  # Aumentado de 0.25 a 0.3
        sentiment = "Positivo"
    elif combined_polarity < -0.3:  # Aumentado de -0.25 a -0.3
        sentiment = "Negativo"
    else:
        sentiment = "Neutral"
    
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
