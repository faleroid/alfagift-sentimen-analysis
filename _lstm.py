import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('./assets/best_model.keras')

with open('./assets/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_LEN = 50

def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    result = model.predict(padded, verbose=0)

    label_index = np.argmax(result)
    score = np.max(result)

    labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    _predict = labels[label_index]
    
    return _predict, score

print("\nLSTM Sentiment Analyzer")

while True:
    input_text = input("\nText: (x for exit)\n")
    if input_text.lower() == 'x':
        break
        
    _predict, score = predict(input_text)
    print(f"Analyzed: {_predict} (Score: {score*100:.1f}%)")