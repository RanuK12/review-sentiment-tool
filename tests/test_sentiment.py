# Forzar actualización del test para GitHub Actions
import unittest
from src.sentiment_model import analyze_sentiment, batch_analyze, get_sentiment_details
import pandas as pd

class TestSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        # Reseñas de hoteles para probar el sistema. ¡Aquí empieza la magia!
        self.hotel_reviews = [
            "The hotel was amazing! The staff was very friendly and the room was clean and comfortable.",
            "Terrible experience. The room was dirty and the service was awful.",
            "It was okay. The room was clean but the breakfast could be better.",
            "Excellent location and great service. Will definitely come back!",
            "The hotel needs renovation. The facilities are outdated.",
            "Perfect stay! Everything exceeded our expectations.",
            "Not worth the price. The room was small and noisy.",
            "The staff was helpful but the room had some maintenance issues.",
            "Beautiful hotel with stunning views. Highly recommended!",
            "Average experience. Nothing special but nothing bad either."
        ]

    def test_positive_reviews(self):
        """¿Detectamos bien las reseñas positivas?"""
        positive_reviews = [
            "The hotel was amazing! The staff was very friendly and the room was clean and comfortable.",
            "Excellent location and great service. Will definitely come back!",
            "Perfect stay! Everything exceeded our expectations.",
            "Beautiful hotel with stunning views. Highly recommended!"
        ]
        for review in positive_reviews:
            polarity, sentiment = analyze_sentiment(review)
            self.assertGreater(polarity, 0.15, f"¡Ojo! No se detectó como positiva: '{review}' (polaridad={polarity})")
            self.assertEqual(sentiment, "Positivo", f"Esperaba 'Positivo' para: '{review}', pero fue '{sentiment}'")

    def test_negative_reviews(self):
        """¿Detectamos bien las reseñas negativas?"""
        negative_reviews = [
            "Terrible experience. The room was dirty and the service was awful.",
            "The hotel needs renovation. The facilities are outdated.",
            "Not worth the price. The room was small and noisy."
        ]
        for review in negative_reviews:
            polarity, sentiment = analyze_sentiment(review)
            self.assertLess(polarity, -0.15, f"¡Ups! No se detectó como negativa: '{review}' (polaridad={polarity})")
            self.assertEqual(sentiment, "Negativo", f"Sentimiento incorrecto para: '{review}'")

    def test_neutral_reviews(self):
        """¿Y las reseñas neutrales? También son importantes."""
        neutral_reviews = [
            "It was okay. The room was clean but the breakfast could be better.",
            "The staff was helpful but the room had some maintenance issues.",
            "Average experience. Nothing special but nothing bad either."
        ]
        for review in neutral_reviews:
            polarity, sentiment = analyze_sentiment(review)
            self.assertTrue(-0.2 <= polarity <= 0.5, f"No se detectó como neutral: '{review}' (polaridad={polarity})")
            self.assertEqual(sentiment, "Neutral", f"Sentimiento incorrecto para: '{review}'")

    def test_batch_analysis(self):
        """¿Funciona el análisis por lotes? ¡Probémoslo con varias reseñas!"""
        df = pd.DataFrame({
            'review': self.hotel_reviews
        })
        result_df = batch_analyze(df)
        # ¿Se analizaron todas las reseñas?
        self.assertEqual(len(result_df), len(self.hotel_reviews), "No se analizaron todas las reseñas.")
        # ¿Existen las columnas nuevas?
        self.assertIn('polarity', result_df.columns, "Falta la columna 'polarity'.")
        self.assertIn('sentiment', result_df.columns, "Falta la columna 'sentiment'.")
        # ¿Todas las polaridades están en el rango correcto?
        self.assertTrue(all(-1 <= p <= 1 for p in result_df['polarity']), "Hay polaridades fuera de rango.")
        # ¿Todos los sentimientos son válidos?
        valid_sentiments = {'Positivo', 'Negativo', 'Neutral'}
        self.assertTrue(all(s in valid_sentiments for s in result_df['sentiment']), "Hay sentimientos no válidos.")

    def test_sentiment_details(self):
        """¿El análisis detallado da toda la info útil?"""
        review = "The hotel was amazing! The staff was very friendly and the room was clean and comfortable."
        details = get_sentiment_details(review)
        # ¿Están todos los campos esperados?
        self.assertIn('polarity', details, "Falta 'polarity' en el análisis detallado.")
        self.assertIn('sentiment', details, "Falta 'sentiment' en el análisis detallado.")
        self.assertIn('positive_keywords', details, "Faltan palabras clave positivas.")
        self.assertIn('negative_keywords', details, "Faltan palabras clave negativas.")
        self.assertIn('confidence', details, "Falta el campo 'confidence'.")
        # ¿Encontramos palabras positivas?
        self.assertGreater(len(details['positive_keywords']), 0, "No se detectaron palabras clave positivas.")
        # ¿La confianza está en el rango correcto?
        self.assertTrue(0 <= details['confidence'] <= 1, "La confianza está fuera de rango.")

if __name__ == '__main__':
    unittest.main() 