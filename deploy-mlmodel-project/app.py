from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('E:/Cloud Drive/OneDrive - University of the Philippines/UPD DCS/3RD YR 2ND SEM/CS 180/cs 180 project/dengue-prediction-web-app/xgb_model.pkl', 'rb'))
  # loading the model
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    features = [float(x) for x in request.form.values()]
    df = pd.DataFrame([features], columns=model.feature_names_in_)
    df.fillna(0)
    prediction = model.predict(df)  # this returns a list e.g. [127.20488798], so pick first element [0]
    label_mapping = {
        4: 'Severe Risk',
        3: 'High Risk',
        2: 'Moderate Risk',
        1: 'Low Risk',
        0: 'Minimal to No risk'
    }
    output = label_mapping[prediction[0]]

    return render_template('index.html', prediction_text=f'Dengue Cases Classification: {output}')

  
if __name__ == "__main__":
    app.run()