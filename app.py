from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
with open("model.pickle", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("home.html")  # HTML form for input


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    input_data = [
        float(data["sepal-length"]),
        float(data["sepal-width"]),
        float(data["petal-length"]),
        float(data["petal-width"]),
    ]
    prediction = model.predict([input_data])
    return jsonify({"prediction": prediction[0]})


if __name__ == "__main__":
    app.run(port=4000, debug=True)
