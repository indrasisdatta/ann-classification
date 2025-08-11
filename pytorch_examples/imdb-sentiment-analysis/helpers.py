def word_to_num(word, vocab):
    if word in vocab:
        return vocab[word]
    else:
        return vocab['<UNK>']