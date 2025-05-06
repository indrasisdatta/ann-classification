import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences 


# Function to predict the next word 
def predict_next_word(model, tokenizer, text, max_sequence_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    # If input is too long, keep last (max_sequence_length-1) words 
    if len(token_list) >= max_sequence_length:
        token_list = token_list[-(max_sequence_length-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    # Convert index back to word
    for word,index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word         
    return None
