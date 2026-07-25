import streamlit as st
import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer
from  nltk.corpus import stopwords

ps = PorterStemmer()

def transform_text(text):
  # lower case
  text = text.lower()
  # tokenize
  text = nltk.word_tokenize(text)

  # Removing special characters
  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
     if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  # Stemming
  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return ' '.join(y)

tfidf = pickle.load(open('verctorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter a message')

if st.button('Predict'):

# 1.Preprocsss
    transformed_text = transform_text(input_sms)
# 2.vectorizer
    vector_input = tfidf.transform([transformed_text])
# 3.Predict
    result = model.predict(vector_input)[0]
# 4.Display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')