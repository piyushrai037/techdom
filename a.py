from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load the trained model
model = tf.keras.models.load_model('sentiment.keras')

# Load the tokenizer
with open('tokenizers.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder
with open('label.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Define the Flask app
app = Flask(__name__)

# Define a route for the API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Preprocess the data
    text = data['text']
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    sequences = tokenizer.texts_to_sequences([tokens])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    # Make prediction
    prediction = model.predict(padded_sequences)

    # Get the class with the highest probability
    predicted_class = np.argmax(prediction)

    # Decode the predicted class
    predicted_label = label_encoder.inverse_transform([predicted_class])

    # Return the result as JSON
    return jsonify(result=predicted_label[0])

if __name__ == '__main__':
    app.run(port=5000, debug=True)