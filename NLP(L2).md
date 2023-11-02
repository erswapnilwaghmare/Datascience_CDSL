### NLP(L2)

1. **Tokenization**:
   - Tokenize a text into individual words.
   ```python
   from nltk.tokenize import word_tokenize
   text = "NLTK is a powerful library for NLP."
   tokens = word_tokenize(text)
   print(tokens)
   ```

   Output:
   ```
   ['NLTK', 'is', 'a', 'powerful', 'library', 'for', 'NLP', '.']
   ```

2. **Stopword Removal**:
   - Remove common stopwords (e.g., "the," "is") from a sentence.
   ```python
   from nltk.corpus import stopwords
   nltk.download('stopwords')
   
   words = word_tokenize("This is an example sentence with stopwords.")
   filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]
   print(filtered_words)
   ```

   Output:
   ```
   ['example', 'sentence', 'stopwords', '.']
   ```

3. **Stemming**:
   - Reduce words to their base or root form (e.g., "jumping" to "jump").
   ```python
   from nltk.stem import PorterStemmer
   stemmer = PorterStemmer()
   word = "jumping"
   stemmed_word = stemmer.stem(word)
   print(stemmed_word)
   ```

   Output:
   ```
   'jump'
   ```

4. **Lemmatization**:
   - Reduce words to their dictionary form (e.g., "jumped" to "jump").
   ```python
   from nltk.stem import WordNetLemmatizer
   nltk.download('wordnet')
   
   lemmatizer = WordNetLemmatizer()
   word = "jumped"
   lemmatized_word = lemmatizer.lemmatize(word, pos='v')
   print(lemmatized_word)
   ```

   Output:
   ```
   'jump'
   ```

5. **POS Tagging**:
   - Assign part-of-speech tags (e.g., noun, verb) to words in a sentence.
   ```python
   from nltk import pos_tag
   words = word_tokenize("The cat is chasing a mouse.")
   tagged_words = pos_tag(words)
   print(tagged_words)
   ```

   Output:
   ```
   [('The', 'DT'), ('cat', 'NN'), ('is', 'VBZ'), ('chasing', 'VBG'), ('a', 'DT'), ('mouse', 'NN'), ('.', '.')]
   ```

6. **Named Entity Recognition (NER)**:
   - Identify and classify named entities (e.g., people, organizations) in a text.
   ```python
   from nltk import ne_chunk
   nltk.download('maxent_ne_chunker')
   nltk.download('words')
   
   text = "Steve Jobs was the co-founder of Apple Inc."
   words = word_tokenize(text)
   tagged_words = pos_tag(words)
   named_entities = ne_chunk(tagged_words)
   print(named_entities)
   ```

   Output:
   ```
   (S
     (PERSON Steve/NNP)
     Jobs/NNP
     was/VBD
     the/DT
     co-founder/NN
     of/IN
     (ORGANIZATION Apple/NNP Inc./NNP)
     ./.)
   ```

7. **Frequency Distribution**:
   - Count the frequency of words in a text and find the most common ones.
   ```python
   from nltk import FreqDist
   words = word_tokenize("This is a simple example. This is another example.")
   fdist = FreqDist(words)
   print(fdist.most_common(2))
   ```

   Output:
   ```
   [('is', 2), ('example', 2)]
   ```

8. **Sentiment Analysis**:
   - Analyze the sentiment (positive, negative, neutral) of a text.
   ```python
   from nltk.sentiment import SentimentIntensityAnalyzer
   nltk.download('vader_lexicon')
   
   analyzer = SentimentIntensityAnalyzer()
   text = "I love this product! It's amazing."
   sentiment_score = analyzer.polarity_scores(text)
   print(sentiment_score)
   ```

   Output:
   ```
   {'neg': 0.0, 'neu': 0.272, 'pos': 0.728, 'compound': 0.6696}
   ```
