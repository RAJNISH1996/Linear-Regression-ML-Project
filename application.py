import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regresor model and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])#standardization of new data
        result=ridge_model.predict(new_data_scaled)#o/p will be in list

        return render_template('home.html',result=result[0])#o/p will be in list,hence we take [0]

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="127.0.0.1",port=8000)
# STEPS TO CONNECT WITH GITHUB    
# ls -a ---to the the association of git hub
# git remote -v ---to see which repositopry need to remove
# git remote rm origin
# now in github make a new repository 
# follow the steps below "create a new repository on the command line"
# git init ---to initialize the repository
# as we already had the readme file so we will do ---git add . ---to make ready all files for git commit
# to check the file tracking --- git status
# git commit -m "first commit" will give error the we need to give the repo id.---git config --global user.email "you@example.com"
# now give your name  ---git config --global user.name "Your Name"
# NOW git commit -m "first commit" to creste mode or we can say ready to commit the all files
# to go to main branch ---git branch -M main
# to see the branch --- git branch
# now provied the address where file is need to store in reo.---git remote add origin https://github.com/RAJNISH1996/Fire-forest-ML-project.git
#  now to send file ---git push -u origin main
# now just follow the coming instructions

# FOR AWS DEPLOYMENT STEP 
# first make configuration file(here is .ebextensions for deploying on benstak)
# create python config file and write code 
# Take care about application:application 
