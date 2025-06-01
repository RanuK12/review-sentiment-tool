import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_reviews
from sentiment_model import batch_analyze

def main():
    df = load_reviews('../data/sample_reviews.csv')
    result = batch_analyze(df)
    print(result[['review', 'sentiment', 'polarity']])

if __name__ == "__main__":
    main()
