from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)
with open("model.pickle", "rb") as f:
    model = pickle.load(f)

port = int(os.environ.get("PORT", 5000))


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
    return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
