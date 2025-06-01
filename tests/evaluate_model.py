import pandas as pd
from src.sentiment_model import analyze_sentiment, batch_analyze, get_sentiment_details
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def load_sample_reviews():
    """
    Carga un conjunto de reseñas de hoteles y viajes para probar el modelo.
    ¡Aquí empieza la aventura de los datos!
    """
    reviews = [
        # Positive Reviews
        "The hotel was amazing! The staff was very friendly and the room was clean and comfortable.",
        "Excellent location and great service. Will definitely come back!",
        "Perfect stay! Everything exceeded our expectations.",
        "Beautiful hotel with stunning views. Highly recommended!",
        "The staff went above and beyond to make our stay memorable.",
        "Great value for money. The amenities were top-notch.",
        "Wonderful experience from check-in to check-out.",
        "The room was spacious and the bed was incredibly comfortable.",
        "Excellent breakfast selection and friendly staff.",
        "The hotel's location was perfect for exploring the city.",
        
        # Negative Reviews
        "Terrible experience. The room was dirty and the service was awful.",
        "The hotel needs renovation. The facilities are outdated.",
        "Not worth the price. The room was small and noisy.",
        "Poor customer service and unresponsive staff.",
        "The room was not cleaned properly during our stay.",
        "Overpriced for what you get. Would not recommend.",
        "The hotel was in a noisy area and the walls were thin.",
        "The air conditioning didn't work properly.",
        "The breakfast was cold and the selection was limited.",
        "The staff was rude and unhelpful.",
        
        # Neutral Reviews
        "It was okay. The room was clean but the breakfast could be better.",
        "The staff was helpful but the room had some maintenance issues.",
        "Average experience. Nothing special but nothing bad either.",
        "The hotel was basic but served its purpose.",
        "The location was convenient but the room was small.",
        "Decent stay for the price we paid.",
        "The hotel was clean but the facilities were limited.",
        "The service was adequate but nothing exceptional.",
        "The room was comfortable but the view was disappointing.",
        "The hotel met our basic needs but lacked character."
    ]
    
    # Create DataFrame with manual sentiment labels
    df = pd.DataFrame({
        'review': reviews,
        'true_sentiment': ['Positivo'] * 10 + ['Negativo'] * 10 + ['Neutral'] * 10
    })
    
    return df

def evaluate_model(df):
    """
    Evalúa el desempeño del modelo de sentimiento con métricas y detalles útiles.
    """
    # Analyze sentiments
    result_df = batch_analyze(df)
    
    # Get detailed analysis for each review
    detailed_analysis = []
    for review in df['review']:
        details = get_sentiment_details(review)
        detailed_analysis.append(details)
    
    # Add detailed analysis to results
    result_df['confidence'] = [d['confidence'] for d in detailed_analysis]
    result_df['positive_keywords'] = [d['positive_keywords'] for d in detailed_analysis]
    result_df['negative_keywords'] = [d['negative_keywords'] for d in detailed_analysis]
    
    # Calculate accuracy
    accuracy = (result_df['sentiment'] == result_df['true_sentiment']).mean()
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(
        result_df['true_sentiment'],
        result_df['sentiment'],
        labels=['Positivo', 'Neutral', 'Negativo']
    )
    
    # Calculate metrics for each sentiment
    metrics = {}
    for sentiment in ['Positivo', 'Negativo', 'Neutral']:
        true_positives = ((result_df['true_sentiment'] == sentiment) & 
                         (result_df['sentiment'] == sentiment)).sum()
        false_positives = ((result_df['true_sentiment'] != sentiment) & 
                          (result_df['sentiment'] == sentiment)).sum()
        false_negatives = ((result_df['true_sentiment'] == sentiment) & 
                          (result_df['sentiment'] != sentiment)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[sentiment] = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score,
            'Support': (result_df['true_sentiment'] == sentiment).sum()
        }
    
    # Calculate average confidence by sentiment
    confidence_by_sentiment = result_df.groupby('sentiment')['confidence'].mean()
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'metrics': metrics,
        'results': result_df,
        'confidence_by_sentiment': confidence_by_sentiment
    }

def plot_results(evaluation_results):
    """
    Genera y guarda gráficos visuales para entender mejor el desempeño del modelo.
    """
    # Configurar el estilo de los gráficos
    plt.style.use('default')
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    cm = evaluation_results['confusion_matrix']
    total = cm.sum()
    cm_percentage = cm.astype('float') / total * 100
    
    sns.heatmap(cm_percentage, 
                annot=True, 
                fmt='.1f', 
                cmap='YlOrRd',
                xticklabels=['Positivo', 'Neutral', 'Negativo'],
                yticklabels=['Positivo', 'Neutral', 'Negativo'])
    plt.title('Matriz de Confusión (Porcentajes)', pad=20, fontsize=14)
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot sentiment distribution
    plt.figure(figsize=(12, 6))
    sentiment_counts = evaluation_results['results']['sentiment'].value_counts()
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # Verde, Amarillo, Rojo
    ax = sentiment_counts.plot(kind='bar', color=colors)
    plt.title('Distribución de Sentimientos Predichos', pad=20, fontsize=14)
    plt.xlabel('Sentimiento', fontsize=12)
    plt.ylabel('Cantidad', fontsize=12)
    
    # Agregar porcentajes sobre las barras
    total = len(evaluation_results['results'])
    for i, v in enumerate(sentiment_counts):
        ax.text(i, v, f'{v}\n({v/total:.1%})', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot polarity distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(evaluation_results['results']['polarity'], 
                 bins=30, 
                 kde=True,
                 color='#3498db')
    plt.title('Distribución de Polaridad de Sentimiento', pad=20, fontsize=14)
    plt.xlabel('Polaridad', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    
    # Agregar líneas verticales para los umbrales
    plt.axvline(x=0.25, color='#e74c3c', linestyle='--', alpha=0.5, label='Umbral Positivo')
    plt.axvline(x=-0.25, color='#e74c3c', linestyle='--', alpha=0.5, label='Umbral Negativo')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('polarity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confidence by sentiment
    plt.figure(figsize=(12, 6))
    confidence_data = evaluation_results['confidence_by_sentiment']
    ax = confidence_data.plot(kind='bar', color=colors)
    plt.title('Confianza Promedio por Sentimiento', pad=20, fontsize=14)
    plt.xlabel('Sentimiento', fontsize=12)
    plt.ylabel('Confianza Promedio', fontsize=12)
    
    # Agregar valores sobre las barras
    for i, v in enumerate(confidence_data):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('confidence_by_sentiment.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load sample reviews
    df = load_sample_reviews()
    
    # Evaluate model
    evaluation_results = evaluate_model(df)
    
    # Print results
    print("\n🌟 Resultados de Evaluación del Modelo:")
    print(f"Precisión General: {evaluation_results['accuracy']:.2%}")
    
    print("\n📊 Matriz de Confusión:")
    conf_matrix = pd.DataFrame(
        evaluation_results['confusion_matrix'],
        index=['Positivo', 'Neutral', 'Negativo'],
        columns=['Positivo', 'Neutral', 'Negativo']
    )
    print(conf_matrix)
    
    print("\n🔎 Métricas Detalladas:")
    for sentiment, metrics in evaluation_results['metrics'].items():
        print(f"\n{sentiment}:")
        for metric, value in metrics.items():
            if metric != 'Support':
                print(f"{metric}: {value:.2%}")
            else:
                print(f"{metric}: {value}")
    
    print("\n📈 Confianza Promedio por Sentimiento:")
    print(evaluation_results['confidence_by_sentiment'].round(3))
    
    # Plot results
    plot_results(evaluation_results)
    print("\n🖼️ Gráficos guardados como:")
    print("- confusion_matrix.png")
    print("- sentiment_distribution.png")
    print("- polarity_distribution.png")
    print("- confidence_by_sentiment.png")

if __name__ == "__main__":
    main() 