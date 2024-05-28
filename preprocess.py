
import pandas as pd
import numpy as np

import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

from nltk.corpus import stopwords

df=pd.read_csv('merged_cleaned2.csv')

def clean_text(txt):
 
    txt  = "".join([char for char in txt if char not in string.punctuation])
    txt = re.sub('[0-9]+', ' ', txt)
    # split into words
    words = word_tokenize(txt)
    
    
    stop_words = set(stopwords.words('greek'))
    words = [w for w in words if not w in stop_words]
    
    
    words = [word for word in words if word.isalpha()]
    words = [word.lower() for word in words]

    cleaned_text = ' '.join(words)
    return cleaned_text
    
df['data_cleaned'] = df['Title'].apply(lambda txt: clean_text(txt))
df.to_csv('stopwords.csv',index=False)

import pandas as pd
import unicodedata as ud


def remove_accents(text):
    if isinstance(text, str):
        d = {ord('\N{COMBINING ACUTE ACCENT}'): None}
        return ud.normalize('NFD', text).translate(d)
    return text


df = pd.read_csv('stopwords.csv')

df['data_cleaned'] = df['data_cleaned'].apply(remove_accents)


df.to_csv('output.csv', index=False)

