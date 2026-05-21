# HANDOFF — Review Sentiment Tool

## Propósito
Herramienta de análisis de sentimientos orientada a reseñas de hoteles y hospitalidad. Combina TextBlob con palabras clave del dominio para clasificar reviews en Positivo / Negativo / Neutral.

## Estado actual
- ✅ Modelo funcional con ajuste de umbrales para polaridad mixta
- ✅ App Streamlit básica operativa
- ✅ Tests unitarios con GitHub Actions
- ✅ Gráficos de evaluación generados

## Stack clave
- Python 3.x
- TextBlob + Pandas
- Streamlit para la UI web
- Scikit-learn para métricas

## Qué funciona
- `analyze_sentiment()` — clasificación individual
- `batch_analyze()` — procesamiento masivo en DataFrame
- `get_sentiment_details()` — análisis con palabras clave detectadas
- Streamlit app en `streamlit_app.py`

## Qué está roto / pendiente
- Streamlit app es muy básica; no usa `get_sentiment_details()` aún
- Falta soporte para español nativo (TextBlob es mejor en inglés)
- No hay API REST ni endpoint para integración

## Próximos pasos
1. Enriquecer la UI de Streamlit con detalles de palabras clave y confianza
2. Agregar soporte multilingüe (spaCy + transformers)
3. Empaquetar como API con FastAPI

## Notas para retomar
- Las palabras clave del dominio están hardcodeadas en `sentiment_model.py`
- Los umbrales de polaridad se ajustaron varias veces; revisar si se necesita recalibrar
- `streamlit run streamlit_app.py` para levantar la app
