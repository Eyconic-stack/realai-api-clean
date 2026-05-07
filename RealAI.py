from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify

texts = [
    "My internet is not working",
    "Laptop is very slow",
    "I need access to my account",
    "Reset my password",
    "System is crashing",
    "Password not working",
    "App is not opening",
    "Computer is freezing",
    "System is crashing",
    "My screen is broken",
    "Keyboard not working"
]

labels = [
    "Technical Issue",
    "Technical Issue",
    "Access Request",
    "Access Request",
    "Technical Issue",
    "Access Request",
    "Unable to sign in",
    "Need access to system",
    "Cannot login to my account",
    "Hardware issue",
    "Hardware issue"
]

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(texts)

model=LogisticRegression()
model.fit(x, labels)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def PREDICT():
    data = request.get_json()
    text = data['text']
    transformed = vectorizer.transform([text])
    result = model.predict(transformed)
    return jsonify({"Prediction": result[0]})



new_ticket = ["My Screen is not showing anything"]
new_x = vectorizer.transform(new_ticket)
prediction = model.predict(new_x)

print(prediction)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)