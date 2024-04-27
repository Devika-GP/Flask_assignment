from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

data  = pd.read_csv(r'Social_Network_Ads.csv')

app = Flask(__name__)

with open('model.pkl','rb') as model_file:
  model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    age = float(request.form["age"])
    est_sal = float(request.form["estsal"])

    features = np.array([[age, est_sal]])
    prediction = model.predict(features)
    target = prediction[0]
    if (target == 0):
        purchase = 'Not purchased'
    elif (target == 1):
        purchase = 'Purchased'
    print(purchase)

    return render_template("index.html", pred_result = purchase)

if __name__ == "__main__":
    app.run(debug=True)
