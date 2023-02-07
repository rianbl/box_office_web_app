# Desenvolvimento de Aplicações Analíticas

# Projeto 2 - Web App Education Analytics Para Previsão de Performance Escolar

# WebApp1

# pip install scikit-learn==0.20
# pip install setuptools --upgrade

# Imports
import numpy as np
import pandas as pd
import sys
import pickle
import joblib
from flask import Flask, render_template, request
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)



# Cria a app
app = Flask(__name__)

# Abre e carrega o modelo
# Load the model from the file
with open("modelo/model_ml.pkl", "rb") as file:
    model_ml = pickle.load(file)

# Load the scaler from the file
with open("modelo/scaler_ml.pkl", "rb") as file:
    scaler_ml = pickle.load(file)

# Load the labelers from the file
with open("modelo/labelers_ml.pkl", "rb") as file:
    labelers_ml = pickle.load(file)

# Load the attributes from the file
with open("modelo/attributes_ml.pkl", "rb") as file:
    attributes_ml = pickle.load(file)


# Rota da página raiz
@app.route('/')
def home():
    return render_template('index.html')

# Rota da página de previsão
@app.route('/predict', methods = ['GET','POST'])
def predict():
    if request.method == 'POST': 
   
        try:

            movie_name = request.form['movie_name']
            int_coprod = request.form['inco']
            fsa = request.form['fsa']
            genre = request.form['genre']
            subtype = request.form['subtype']
            distributor = request.form['distributor']
            director = request.form['director']
            screenwritter = request.form['screenwritter']
            movie_length = request.form['movie_length']
            year = request.form['year']
            start_date = request.form['start_date']



            # Coloca todas as variáveis em uma lista e ajusta o shape
            pred_args = [int_coprod, fsa, genre, subtype, distributor, director, screenwritter, movie_length, year, start_date]
            pred_args_arr = pd.DataFrame(pred_args).T
            pred_args_arr.columns = attributes_ml
            
            labeled_arr = pred_args_arr.copy()
            for key in list(labelers_ml.keys()):
                labeled_arr[key] = labelers_ml[key].transform(labeled_arr[key])

          
            scaled_arr = scaler_ml.transform(labeled_arr)

           
            # Faz as previsões

            model_prediction = round(model_ml.predict(scaled_arr)[0])
            model_prediction = "{:,}".format(model_prediction)

        except:
            return "Make sure you have input the correct values!"
   
    return render_template('predict.html', prediction = model_prediction, movie_name = movie_name)        

# Executa a app
if __name__ == '__main__':
    app.run()  


