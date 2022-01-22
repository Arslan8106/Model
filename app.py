import numpy as np
from flask import Flask, request, render_template
import pandas as pd


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    '''
    For direct API calls trought request
    '''
    col_names = ['Age','Height', 'Weight', 'BMR', 'PAL', 'TEE', 'BMI','Disease', 'Diet_Plan']
    dataset = pd.read_csv("Female diseases.csv",header=None, names=col_names)
    dataset
    feature_cols = ['Age', 'Height', 'Weight', 'BMR', 'PAL', 'TEE', 'BMI', 'Disease']
    X = dataset[feature_cols]
    y = dataset.Diet_Plan
    def convert_to_int(word):
      word_dict={'DB':1, 'HY':2, 'CP':3, 'CV':4}
      return word_dict[word]
      
    X['Disease'] = X['Disease'].apply(lambda x : convert_to_int(x))
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred  =  classifier.predict(X_test)
    y_pred  

    if request.method == 'POST':
        feature_cols = ['Age', 'Height', 'Weight', 'BMR', 'PAL', 'TEE', 'BMI','Disease']
        X = dataset[feature_cols]
        y = dataset.Diet_Plan

        X['Disease'] = X['Disease'].apply(lambda x : convert_to_int(x))

        classifier = GaussianNB()
        classifier.fit(X, y)
        # age = request.form.get("age")
        # height = request.form.get("height")
        # weight = request.form.get("weight")
        # bmr = request.form.get("bmr")
        # pal = request.form.get("pal")
        # tee = request.form.get("tee")
        # bmi = request.form.get("bmi")
        # disease = request.form.get("disease")
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        predicted_dietplan = classifier.predict(final_features)
        output = predicted_dietplan[0]

    return render_template('index.html', prediction_text='Predicted diet plan is $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)