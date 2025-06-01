import pytest
from textblob import TextBlob

def test_positive_sentiment():
    """Test that positive text is correctly identified."""
    text = "I absolutely loved my stay at this hotel! The service was excellent and the rooms were beautiful."
    blob = TextBlob(text)
    assert blob.sentiment.polarity > 0.1

def test_negative_sentiment():
    """Test that negative text is correctly identified."""
    text = "The hotel was terrible. Dirty rooms and rude staff. Would not recommend."
    blob = TextBlob(text)
    assert blob.sentiment.polarity < -0.1

def test_neutral_sentiment():
    """Test that neutral text is correctly identified."""
    text = "The hotel is located in the city center. The check-in process was standard."
    blob = TextBlob(text)
    assert -0.1 <= blob.sentiment.polarity <= 0.1

def test_sentiment_polarity_range():
    """Test that sentiment polarity is always between -1 and 1."""
    texts = [
        "This is the best experience ever!",
        "This is the worst experience ever!",
        "The hotel is okay.",
        "I don't know what to think about this place."
    ]
    
    for text in texts:
        blob = TextBlob(text)
        assert -1 <= blob.sentiment.polarity <= 1
        assert 0 <= blob.sentiment.subjectivity <= 1

def test_empty_text():
    """Test handling of empty text."""
    text = ""
    blob = TextBlob(text)
    assert blob.sentiment.polarity == 0.0
    assert blob.sentiment.subjectivity == 0.0

def test_special_characters():
    """Test handling of text with special characters."""
    texts = [
        "Great hotel! 👍",
        "Terrible experience 😡",
        "Average stay...",
        "Hotel was okay (but expensive)"
    ]
    
    for text in texts:
        blob = TextBlob(text)
        assert -1 <= blob.sentiment.polarity <= 1

def test_multilingual_text():
    """Test handling of multilingual text."""
    texts = [
        "Excelente hotel! The service was great!",
        "Muy mal servicio. Terrible experience.",
        "Hotel normal. Nothing special."
    ]
    
    for text in texts:
        blob = TextBlob(text)
        assert -1 <= blob.sentiment.polarity <= 1

def test_long_text():
    """Test handling of long text."""
    text = """
    I stayed at this hotel for a week during my business trip. The location was perfect,
    right in the center of the city. The staff was friendly and helpful, always ready to
    assist with any request. The rooms were clean and comfortable, though a bit small.
    The breakfast buffet was excellent with a good variety of options. The only downside
    was the slow WiFi connection in the rooms. Overall, it was a pleasant stay and I
    would recommend it to others.
    """
    blob = TextBlob(text)
    assert -1 <= blob.sentiment.polarity <= 1
    assert 0 <= blob.sentiment.subjectivity <= 1

def test_numbers_and_symbols():
    """Test handling of text with numbers and symbols."""
    texts = [
        "5-star hotel experience!",
        "2/10 would not recommend",
        "Hotel rating: 4.5/5",
        "Price: $200/night - worth it!"
    ]
    
    for text in texts:
        blob = TextBlob(text)
        assert -1 <= blob.sentiment.polarity <= 1

def test_whitespace_handling():
    """Test handling of text with various whitespace patterns."""
    texts = [
        "   Great hotel!   ",
        "\tExcellent service\n",
        "  Good  experience  ",
        "Bad\nservice\t"
    ]
    
    for text in texts:
        blob = TextBlob(text)
        assert -1 <= blob.sentiment.polarity <= 1 