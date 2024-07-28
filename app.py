from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')

@app.route('/p2')
def p2():
    return render_template('home.html')


@app.route('/p3')
def p3():
    return render_template('p3.html')

@app.route('/predict', methods=['POST'])
def p22():
    uname = request.form['username']
    fever = request.form['Fever']
    tiredness = request.form['Tiredness']
    dry_Cough = request.form['Dry-Cough']
    diff_in = request.form['Difficulty-in-Breathing']
    sore = request.form['Sore-Throat']
    nonesy = request.form['None_Sympton']
    pains = request.form['Pains']
    nasal = request.form['Nasal-Congestion']
    runny = request.form['Runny-Nose']
    dia = request.form['Diarrhea']
    noneex = request.form['None_Experiencing']
    gender = request.form['Gender_Male']
    cont = request.form['Contact_Yes']
    arr = np.array([[fever,tiredness,dry_Cough,diff_in,sore,nonesy,pains,nasal,runny,dia,noneex,gender,cont,]])
    pred = model.predict(arr)
    scaletemp = 0
    if(pred == ['Severe']) :
        scaletemp = 4
    elif(pred == ['Moderate']):
        scaletemp = 3
    elif(pred == ['None']):
        scaletemp = 1
    elif(pred == ['Mild']):
        scaletemp = 2
    return render_template('p3.html', data = pred, Name = uname, scale = scaletemp)


if __name__ == "__main__":
    app.run(debug=True)