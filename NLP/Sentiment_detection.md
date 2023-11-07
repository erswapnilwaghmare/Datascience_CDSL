# Sentiment Detection
## Without any packages

```python
def detect_sentiment(text):
    # Define simple keyword lists for positive, negative, and neutral sentiments
    positive_keywords = ['happy', 'joyful', 'excited', 'glad', 'delighted']
    negative_keywords = ['sad', 'unhappy', 'depressed', 'miserable', 'sorrowful']
    neutral_keywords = ['okay', 'fine', 'alright', 'neutral']

    # Normalize the text to lower case for comparison
    text = text.lower()

    # Check for sentiment keywords in the text
    for word in positive_keywords:
        if word in text:
            return 'Happy'
    for word in negative_keywords:
        if word in text:
            return 'Sad'
    for word in neutral_keywords:
        if word in text:
            return 'Neutral'
    
    # If no keywords found, return neutral as the default sentiment
    return 'Neutral'

# Test the function
text = input("Enter your text: ")
sentiment = detect_sentiment(text)
print(sentiment)
```

## Implementing an NLP algorithm for Sentiments Detection
```python
from textblob import TextBlob

def detect_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    # Obtain the sentiment of the text
    sentiment = blob.sentiment

    # polarity is a float within the range [-1.0, 1.0]
    # subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective
    if sentiment.polarity > 0:
        return 'Positive'
    elif sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Test the function
text = input("Enter your text: ")
sentiment = detect_sentiment(text)
print(sentiment)

```
