from flask import Flask, request
from flask_cors import CORS
import validator
import swdantic

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
    print(request.args)
    try:
        return {"message":{"prediction":{"is_resistance":v.run(request.args)}}}, 200
    except Exception as err:
        return "prediction failed" + str(err), 500

@app.route("/features",methods=["GET"])
def features():
    return v.getFeatures(), 200

if __name__ == '__main__':
    v = validator.Validator(proc_pkl="pkl/ptransformer.pkl",
                            model_pkl="pkl/model.pkl",
                            bmodel=swdantic.SWDantic,
                            y_col="empire_or_resistance_resistance")
    app.run()
    # app.run(debug=True)