import pickle
import sklearn
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)#starting point
#Intialise Model
model = pickle.load(open('randfc.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')#creates welcome page
def home():#This Render the front page
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])#creates model prediction ability
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():#GET VALUES FROM USER AND MAKE PREDICTION
    data=[float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = model.predict(final_input)[0]
    quality = {0: 'poor', 1: 'good', 2: 'excellent'}
    return render_template('home.html', prediction_text=f'The wine that will be produced will be of {quality[output]} quality')
if __name__ == '__main__':
    app.run(debug=True)


