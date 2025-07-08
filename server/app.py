from flask import Flask, request
from flask_cors import CORS
import validator

app = Flask(__name__)
CORS(app)

@app.route("/help", methods=['GET'])
def help():
    return """Valid endpoints include:
- GET /ping
- POST /predict
- GET /features
- GET /explain"""

@app.route("/ping", methods=['GET'])
def ping():
    return "pong", 200

@app.route("/predict",methods=["POST"])
def predict():
    try:
        return {"message":{"prediction":{"Frequent_Losers":v.run(request.args)}}}, 200
    except Exception as err:
        return "prediction failed" + str(err), 500

if __name__ == '__main__':
    v = validator.Validator("scaler.pkl","model.pkl")
    app.run()
    # app.run(debug=True)