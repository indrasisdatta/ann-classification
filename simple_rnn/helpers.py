from tensorflow.keras.preprocessing import sequence 

#Helper functions 

# Function to decode reviews 
def decode_reviews(reverse_word_index, encoded_reviews):
    return ' '.join([reverse_word_index.get(i-3) for i in encoded_reviews])

# Function to preprocess user input 
def preprocess_text(word_index, text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review 

# Prediction function 
def predict_sentiment(model, word_index, review):
    preprocessed_input = preprocess_text(word_index, review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]