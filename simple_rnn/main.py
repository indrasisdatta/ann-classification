# Load libraries
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import load_model 
import streamlit as st
from helpers import predict_sentiment 

# Load the IMDB dataset word index 
word_index = imdb.get_word_index()
reverse_word_index = {val:key for key, val in word_index.items()}

# Load the Pretrained model with Relu activation 
model = load_model('simple_rnn_imdb.h5')

# Streamlit
st.title('IMDB Movie review sentiment analysis') 
st.write('Enter a movie to classify as Positive or Negative')

# User input 
user_input = st.text_area("Movie Review")

if st.button('Classify'):
    sentiment, prediction_score = predict_sentiment(model, word_index, user_input)
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Score: {prediction_score}')
else :
    st.write('Please enter a movie review')
