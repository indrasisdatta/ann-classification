import os
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report 

def handle_submit(preprocessing, word_embedding, ml_algo):
    # Load data from CSV
    df = pd.read_csv('spam-classifier/spam.csv', encoding='latin-1')
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    df.drop_duplicates(inplace=True)
    df.rename(columns= {'v1': 'Category', 'v2': 'Message'}, inplace=True)
    df['Message'] = df['Message'].str.lower().str.strip()

    # Apply label encoding to Category column (0 - Spam, 1 - ham)
    le = LabelEncoder()
    y = le.fit_transform(df['Category'])

    inputs = df.iloc[:,1]
    sentence = []
    if preprocessing == "Enable":
        sentence = preprocess_sentence(inputs)
    else:
        sentence = df['Message']

    if word_embedding == "TF-IDF":
        tfidf = TfidfVectorizer(ngram_range=(1,2))
        X = tfidf.fit_transform(sentence)
    elif word_embedding == "BOW":       
        cv = CountVectorizer(ngram_range=(1,2))
        X = cv.fit_transform(sentence)

    return model_regression_metrics(X, y, ml_algo)    
    

def preprocess_sentence(inputs): 
    stopwords_list = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    sentence = []
    for input in inputs:
        # Convert sentence to words using Word Tokenizer
        words = nltk.word_tokenize(input)
        words_list = []
        # Remove the words which are present in stopwords list 
        # Lemmatize words and store in sentence variable
        for word in words:
            if word not in stopwords_list:
                words_list.append(lemmatizer.lemmatize(word))
        sentence.append(' '.join(words_list))

    return sentence

def model_regression_metrics(X, y, ml_algo):
    match ml_algo:
        case 'Logistic Regression':
            model = LogisticRegression(max_iter=1000, random_state=0)
        case 'Random Forest':
            model = RandomForestClassifier(random_state=0) 
        case 'Naive Bayes':
            model = MultinomialNB()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model.fit(X_train, y_train) 
    
    y_pred = model.predict(X_test)

    return classification_report(y_test, y_pred, output_dict=True)





    


