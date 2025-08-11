import torch
from helpers import prediction
# import streamlit as st 

# st.set_page_config(page_title="Autocomplete")
# st.title("Autocomplete")
# st.text("Next Word Predictor")

# st.selectbox("Enter your search text", ["Hi", "Hi, how", "Hi, how are"])

input_text = "order"
suggestions = []        
for i in range(5):
    output_text = prediction(input_text)
    if output_text is not None:
        input_text = output_text
        suggestions.append(output_text)
    else: 
        break

print(suggestions)





