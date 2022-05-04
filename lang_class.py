# NLP model classifying the language a sentence belongs to
# Dataset from kaggle: https://www.kaggle.com/datasets/basilb2s/language-detection?resource=download

import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

df = pd.read_csv('Language Detection.csv')

# print(df.head())

# 17 possible languages to choose from
# print(df['Language'].nunique())
# print(df['Language'].unique())

"""
DATA PROCESSING
"""


# Eliminating stopwords, punctuation, & numbers from sentences
def text_process(mess):
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english') and not word.isdigit()]


# Converting languages to dummy variables
# lang_dummies = pd.get_dummies(df['Language'], drop_first=True)
# df = pd.concat([df.drop('Language', axis=1), lang_dummies], axis=1)
# print(df.head())
df['LanguageNum'] = df['Language'].replace(['English', 'Malayalam', 'Hindi', 'Tamil', 'Portugeese', 'French', 'Dutch',
                                            'Spanish', 'Greek', 'Russian', 'Danish', 'Italian', 'Turkish', 'Sweedish',
                                            'Arabic', 'German', 'Kannada'],
                                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

# print(df.head())

# String to token integer values
bow_transformer = CountVectorizer(analyzer=text_process).fit(df['Text'])

# Tokenizing all of the messages in the dataframe
messages_bow = bow_transformer.transform(df['Text'])

# Using tfidf to assign weights to words in messages
tfidf_transformer = TfidfTransformer().fit(messages_bow)

# Converting entire corpus into a tfidf corpus
messages_tfidf = tfidf_transformer.transform(messages_bow)

# Utilizing the naive bayes classifier
spam_detect_model = MultinomialNB().fit(messages_tfidf, df['Text'])

msg_train, msg_test, label_train, label_test = train_test_split(df['Text'], df['LanguageNum'], test_size=0.3)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(label_test, predictions))

