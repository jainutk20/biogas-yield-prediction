from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model('./model.h5')

app = Flask(__name__)
cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
  # Get the data from the POST request.
  data = request.get_json(force=True)
  print(data)
  # Make prediction using the model
  prediction = model.predict(data)
  print(prediction)

  # Return the prediction as a JSON object
  response = jsonify(str(prediction[0][0]))
  return response

if __name__ == '__main__':
  app.run(debug=True)