import streamlit as st 
from helpers import handle_submit

st.title("NLP Spam Classifier")
st.text("Select options for NLP Spam Classifier Performance Metrics")

# Form inputs
preprocessing = st.selectbox(
    "Pre-processing",
    ("Disable", "Enable")
)
word_embedding = st.selectbox(
    "Select Word Embedding",
    ("BOW", "TF-IDF")
)
ml_algo = st.selectbox(
    "Select Algorithm",
    ("Logistic Regression", "Random Forest", "Naive Bayes")
)
submit_btn = st.button("Submit")

if submit_btn:
    # Form validation
    if not preprocessing:
        st.error("Please select Pre-processing option")
    elif word_embedding == None:  
        st.error("Please select Word Embedding")
    elif not ml_algo:
        st.error("Please select ML Algorithm")
    # Valid form, process next steps
    else: 
        with st.spinner():
            logistic_regression_metrics = handle_submit(preprocessing, word_embedding, ml_algo)
            # Display metrics output 
            st.dataframe(logistic_regression_metrics)


