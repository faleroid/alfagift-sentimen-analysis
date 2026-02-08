import joblib
import re

loaded_tfidf = joblib.load('./assets/tf_idf.joblib')
loaded_rf = joblib.load('./assets/rf_tfidf.joblib')

label_map = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def predict(text):
    clean_input = clean_text(text)
    vec_input = loaded_tfidf.transform([clean_input])
    
    pred_index = loaded_rf.predict(vec_input)[0]
    pred_proba = loaded_rf.predict_proba(vec_input)[0]
    
    result = label_map[pred_index]
    score = pred_proba[pred_index] * 100
    
    return result, score

print("\nRF TF-IDF Sentiment Analyzer")

while True:
    txt = input("\nText: (x for exit)\n")
    if txt == 'x': break
    
    result, score = predict(txt)
    print(f"Analyzed: {result} (Score: {score:.1f}%)")