### NLTK's sentiment analysis:

| Term       | Description                                      | Example Score | Example Sentence                            |
|------------|--------------------------------------------------|---------------|---------------------------------------------|
| 'neg'      | The negative sentiment score. It represents the intensity of negative sentiment in the text, ranging from 0.0 (no negative sentiment) to 1.0 (extremely negative). | 'neg': 0.0 | "I love sunny days at the beach." |
| 'neu'      | The neutral sentiment score. It represents the intensity of neutral sentiment in the text, ranging from 0.0 (no neutral sentiment) to 1.0 (completely neutral). | 'neu': 0.226 | "The sky is blue, and the weather is pleasant." |
| 'pos'      | The positive sentiment score. It represents the intensity of positive sentiment in the text, ranging from 0.0 (no positive sentiment) to 1.0 (extremely positive). | 'pos': 0.774 | "The party was amazing, and I had a great time!" |
| 'compound' | The compound sentiment score. It provides an overall sentiment score that considers the combined effect of positive, negative, and neutral sentiments. It ranges from -1.0 (extremely negative) to 1.0 (extremely positive). A score close to 1.0 indicates a very positive sentiment, while a score close to -1.0 indicates a very negative sentiment. A score near 0.0 suggests a neutral or mixed sentiment. | 'compound': 0.8402 | "I am absolutely thrilled with the fantastic news! It's a dream come true." |

Example:
```python
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()

# Example Sentences
sentence1 = "I love sunny days at the beach."
sentence2 = "The sky is blue, and the weather is pleasant."
sentence3 = "The party was amazing, and I had a great time!"
sentence4 = "I am absolutely thrilled with the fantastic news! It's a dream come true."

# Sentiment Analysis
sentiment_score1 = analyzer.polarity_scores(sentence1)
sentiment_score2 = analyzer.polarity_scores(sentence2)
sentiment_score3 = analyzer.polarity_scores(sentence3)
sentiment_score4 = analyzer.polarity_scores(sentence4)

print(sentiment_score1)
print(sentiment_score2)
print(sentiment_score3)
print(sentiment_score4)
```

### NLTK Terms


| Term                           | Explanation                                                                      | Example                                       |
|--------------------------------|----------------------------------------------------------------------------------|-----------------------------------------------|
| Tokenization                    | Breaking down text into individual words or tokens.                              | "NLTK is powerful" → ["NLTK", "is", "powerful"] |
| Stopwords                       | Common words (e.g., "the," "is") often removed from text due to low significance.| "The quick brown fox" → ["quick", "brown", "fox"] |
| Stemming                        | Reducing words to their base or root form by removing suffixes.                 | "running" → "run"                            |
| Lemmatization                   | Reducing words to their dictionary or base form while considering meaning.       | "better" → "good"                            |
| Part-of-Speech (POS) Tagging    | Assigning grammatical tags (e.g., noun, verb) to words in a sentence.            | "The cat jumps." → [("The", "DT"), ("cat", "NN"), ("jumps", "VB")] |
| Named Entity Recognition (NER)  | Identifying and classifying named entities (e.g., names, organizations) in text.  | "Steve Jobs founded Apple Inc." → [("Steve Jobs", "PERSON"), ("Apple Inc.", "ORG")] |
| Frequency Distribution          | Counting word frequencies in text and identifying common words.                   | "This is a simple example." → {"This": 1, "is": 1, "a": 1, "simple": 1, "example": 1} |
| Sentiment Analysis              | Determining the emotional tone of text (e.g., positive, negative, neutral).      | "I love NLTK. It's great!" → {'neg': 0.0, 'neu': 0.188, 'pos': 0.812, 'compound': 0.8402} |
| Corpus                          | A collection of text documents used for linguistic analysis and research.        | A collection of news articles.              |
| WordNet                        | A lexical database providing word relationships and meanings.                     | Finding synonyms for "happy."                |
| NLTK                           | The Natural Language Toolkit, a Python library for language processing.         | Importing and using NLTK in Python.         |
| VADER Sentiment Analysis       | A rule-based sentiment analysis tool used in NLTK.                               | Analyzing sentiment using VADER in NLTK.    |
| N-grams                        | Sequences of n words (e.g., bigrams, trigrams) used in text analysis.            | "The quick brown fox" → bigrams: [("The", "quick"), ("quick", "brown"), ...] |
| Chunking                       | Grouping words into chunks based on grammatical structure.                        | "She quickly eats an apple" → Chunk: [("She", "NP"), ("quickly eats", "VP"), ("an apple", "NP")] |
| Syntax                         | The structure and arrangement of words and phrases in a sentence.                | Understanding sentence grammatical structure. |
| Morphology                     | The study of word structure and formation.                                       | Analyzing word prefixes and suffixes.        |
| Semantics                      | The study of meaning in language and how words convey meaning.                    | Understanding the meaning of "run" in different contexts. |
| Syntax Tree                    | A graphical representation of sentence syntactic structure.                       | Diagram showing sentence structure.           |
| Bag of Words (BoW)            | Text representation as a collection of words, ignoring word order.                | Creating a BoW model for text classification. |
| TF-IDF                         | A numerical representation of word importance in a document collection.          | Calculating TF-IDF scores for document keywords. |


### Example:

```Python
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

def analyze_sentiment(input_text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(input_text)
    
    # Extract positive and negative sentiment scores
    positive_score = sentiment_score['pos'] * 100
    negative_score = sentiment_score['neg'] * 100
    
    print("Sentiment Analysis:")
    print(f"Positive: {positive_score:.2f}%")
    print(f"Negative: {negative_score:.2f}%")

# Input text
input_text = "I love that toy."

# Analyze sentiment
analyze_sentiment(input_text)


```