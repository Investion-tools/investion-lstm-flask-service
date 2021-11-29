from flask import Flask, jsonify, request
import LSTM

app = Flask(__name__)

@app.route("/make-prediction/<asset>")
def make_prediction(asset):
    return jsonify(LSTM.predict_tomorrow_price(asset))

if __name__ == "__main__":
    app.run(debug=True)