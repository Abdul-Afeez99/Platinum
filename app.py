from flask import Flask, render_template, request
import joblib
import string
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)

model = joblib.load('news_review_model.pkl')

vectorizer = joblib.load('fitted_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        news_title = request.form['news_title']
        url = request.form['news_url']
        source_domain = request.form['source_domain']
        user_input = news_title + " " + url + " " + source_domain
        preprocessed_input = preprocess_text(user_input)

        text_input_transformed = vectorizer.transform([preprocessed_input])
        text_input_array = text_input_transformed.toarray()

        prediction = model.predict(text_input_array)[0]

        return render_template('result.html', prediction=prediction)

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

if __name__ == '__main__':
    app.run(debug=True)
