from flask import Flask,render_template,request,redirect,url_for,session
import pickle 
import numpy as np

app = Flask(__name__)
model= pickle.load(open("model_mode.pkl","rb"))
model2 = pickle.load(open("model_date.pkl","rb"))


@app.route('/')
def hello_world(): 
    return render_template('index.html')


@app.route('/predictdate',methods=['GET','POST'])
def predictdate():
    if request.method == 'GET':
        return render_template('temp.html')
    else:
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = model2.predict(features)
        prediction = prediction.astype(int)
        return render_template('temp.html',prediction = prediction)


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html',prediction = 'NULL')
    else:
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = model.predict(features)

        if prediction[0] == 0:
            mode = 'Air'
        elif prediction[0] == 1:
            mode = 'Sea'    
        elif prediction[0] == 2:
            mode = 'Truck'
        else:
            mode = 'Charter Plane'    

        return render_template('predict.html',prediction = mode)
        

# main driver function
if __name__ == '__main__':
 
    app.run(debug=True)