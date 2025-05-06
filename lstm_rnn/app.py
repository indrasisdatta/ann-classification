import streamlit as st 
import numpy as np  
import pickle 
import tensorflow as tf 
from tensorflow.keras.models import load_model
from helper import predict_next_word

# Load the LSTM model 
model = load_model('next_word_lstm_model.h5')

# Load the tokenizer 
with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

# Streamlit app 
st.title('Next word prediction')
input_text = st.text_input("Enter the sequence of words", "To be or not to be")
if st.button("Predict next word"):
    max_sequence_length = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
    st.write("Next word: " + next_word)


