from sympy import content
import torch 
import torch.nn as nn 
import numpy as np 
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize 
import nltk 

# Convert the entire sentence to vocab numbers 
def word_to_num(word, vocab):
    if word in vocab:
        return vocab[word]
    else: 
        return vocab['<unk>']
    
# Form sequences 
# [1,2,3] -> [1,2] [1,2,3]
# [3,5,1,2] -> [3,5] [3,5,1] [3,5,1,2]

def form_sequences(numbers):
    limit = len(numbers)
    if limit <= 1:
        return numbers
    sequences = []
    for i in range(1, limit):        
        sequences.append(numbers[:i+1])
    return sequences

class CustomDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X 
        self.y = y

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__() 
        self.embedding = nn.Embedding(vocab_size, 100) 
        self.lstm = nn.LSTM(100, 150, batch_first=True)
        self.fc = nn.Linear(150, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        intermediate_hidden_state, (final_hidden_state, final_cell_state) = self.lstm(embedded)
        output = self.fc(final_hidden_state.squeeze(0))
        return output
    
def prediction(text):

    checkpoint = torch.load('next_word_model.pth')
    vocab = checkpoint['vocab']
    acceptable_len = checkpoint['acceptable_len']

    model = LSTMModel(len(checkpoint['vocab']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Tokenize text 
    tokenized_text = word_tokenize(text)

    # Text to numerical indices 
    numerical_text = []
    for word in tokenized_text:
        numerical_text.append(word_to_num(word, vocab))

    # Padding 
    # print(numerical_text)
    padded_text = [0]*(acceptable_len - len(numerical_text)) + numerical_text
    padded_text = torch.tensor(padded_text, dtype=torch.long).unsqueeze(0)
    
    # Send to model 
    output = model(padded_text)

    # Predicted index 
    value, index = torch.max(output, dim=1)
    print("Next word prediction -->", value, index)
    if value == '<UNK>':
        return None
    
    # Merge with text
    text += " " + list(vocab.keys())[index]
    return text

