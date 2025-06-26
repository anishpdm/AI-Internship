from flask import Flask

app = Flask(__name__) 


#API
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/predict")
def predict():
    return "AI prediction"

if __name__ == '__main__':
    app.run(debug=True)

