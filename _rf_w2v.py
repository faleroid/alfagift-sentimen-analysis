import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import re

rf_model = joblib.load('./assets/rf_w2v.joblib')
w2v_model = Word2Vec.load("./assets/word2vec.model")

label_map = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

def text_cleaning(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.split()

def get_avg_vector(tokens, model):
    vector_size = model.vector_size
    vec = np.zeros(vector_size)
    count = 0
    
    for word in tokens:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    
    if count > 0:
        vec /= count
        
    return vec.reshape(1, -1)

def predict(teks_baru):
    tokens = text_cleaning(teks_baru)
    
    if len(tokens) == 0:
        return "Not Valid Text"
        
    vector_input = get_avg_vector(tokens, w2v_model)
    
    pred_index = rf_model.predict(vector_input)[0]
    pred_proba = rf_model.predict_proba(vector_input)[0]
    
    result = label_map[pred_index]
    confidence = np.max(pred_proba) * 100
    
    return result, confidence

print("\nRF Word2Vec Sentiment Analyzer")

while True:
    input_text = input("\nText: (x for exit)\n")
    if input_text.lower() == 'x': break
    
    result, confidence = predict(input_text)
    print(f"Analyzed: {result} (Score: {confidence:.1f}%)")