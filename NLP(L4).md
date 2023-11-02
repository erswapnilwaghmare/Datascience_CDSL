```python

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# Download the required NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Input text
input_text = "Steve"

# Tokenize the text and perform Part-of-Speech (POS) tagging
tokens = word_tokenize(input_text)
pos_tags = pos_tag(tokens)

# Perform Named Entity Recognition (NER) with custom rules
custom_entities = []

for word, pos in pos_tags:
    if pos == 'NNP':
        custom_entities.append((word, 'PERSON'))
    else:
        custom_entities.append((word, 'O'))  # 'O' for other

print("Custom Named Entities:")
print(custom_entities)

```