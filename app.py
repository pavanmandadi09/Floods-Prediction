import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('model_k.pkl')

@app.route('/')
def home() :
    return render_template('home.html')
@app.route('/Predict')
def index() :
    return render_template("index.html")



@app.route('/predict', methods = ['POST'])
def predict() :
    input_data = {
        'Temperature' : request.form.get('T'),
        'Humidity' : request.form.get("H"),
        'Cloud_cover' : request.form.get("C"),
        'Annual_rainfall' : request.form.get('A'),
        'Jan-Feb_rainfall' : request.form.get("J"),
        'Mar-May_rainfall' : request.form.get('M'),
        'Jun-Sep_rainfall' : request.form.get("S"),
        'Oct-Dec_rainfall' : request.form.get('O'),
        'Avg_june_rainfall' : request.form.get('AJ'),
        'Sub_surafce_water_level' : request.form.get('SS')
    }

    df = pd.DataFrame([input_data])

    for col in model.feature_names_in_ :
        if col not in df.columns :
            df[col] = 0
    
    df = df[model.feature_names_in_]

    prediction = model.predict(df)[0]

    if(prediction == 0) :
        return render_template('nosevere.html',prediction = 'No possibility of severe flood')
    else :
        return render_template('severe.html',prediction = 'Possibility of Severe Flood')




# @app.route('/back')
# def back() :
#     return render_template('index.html')
# @app.route('/exit')
# def exit():
#     return render_template('exit.html')

if __name__ == '__main__' :
    app.run(debug=True)


    
