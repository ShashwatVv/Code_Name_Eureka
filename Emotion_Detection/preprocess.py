import re
import spacy
import string
from textblob import TextBlob


en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words


def preprocessing(text):
    '''
        1) lowercase
        2) url removal
        3) punctuations rem

        4) stopwords rem
        5) spelling correction

        5) lemmatization

    '''

    # lowercase
    text = text.lower()

    # remove url
    text = re.sub(r'http\S+', '', text)

    # remove punctuations
    text = text.translate(
        str.maketrans('', '', string.punctuation))  # map the characters in string.punctuations to none

    # spelling correction
    text = str(TextBlob(text).correct())

    # stopwords removal and tokenization and lemmatizatin
    text = en(text)
    res_tokens = []
    for token in text:
        if not token.is_stop:
            res_tokens.append(token.lemma_)

    return ' '.join(res_tokens)

