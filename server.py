from flask import Flask, render_template, request
import joblib


app = Flask(__name__)

vectorizer = joblib.load("my_model/vectorizer.pkl")
opinion_model = joblib.load("my_model/sentiment_model.pkl")

 
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/score", methods=['GET', 'POST'])
def score():
    if request.method == 'POST':
        text = request.form['ftext']
        new_text = vectorizer.transform([text])
        new_text_pred = opinion_model.predict(new_text)[0]
        new_text_prob = opinion_model.predict_proba(new_text)[0][1]   
        return render_template("score.html", text=text, pred=new_text_pred, prob=new_text_prob)
    else:
        return render_template("score.html")

if __name__ == '__main__':
    app.run(debug=True)