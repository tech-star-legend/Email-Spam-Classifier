import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):

    # Lowercase conversion
    text = text.lower()

    # Tokenization
    words = nltk.word_tokenize(text)

    filtered_words = []

    # Remove special characters
    for word in words:
        if word.isalnum():
            filtered_words.append(word)

    temp = filtered_words[:]
    filtered_words.clear()

    # Remove stopwords and punctuation
    for word in temp:
        if word not in stopwords.words('english') and word not in string.punctuation:
            filtered_words.append(word)

    temp = filtered_words[:]
    filtered_words.clear()

    # Stemming
    for word in temp:
        filtered_words.append(ps.stem(word))

    return " ".join(filtered_words)