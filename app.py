from flask import Flask, request, json
import iris


app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/predict", methods=['POST'])
def predict():
    data = json.loads(request.data)
    result = iris.classifier.predict([[data['sl'], data['sw'], data['pl'], data['pw']]])[0]
    print(result)
    return json.dumps({'iris_varian' : iris.lb.classes_[result]})