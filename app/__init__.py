from flask import Flask, request, jsonify
from flask_cors import CORS
from .new_model_pred import new_model_pred
from .old_model_pred import old_model_pred
import os

# Initialize Flask application
app = Flask(__name__)
CORS(app)

@app.route("/prediction", methods=["POST"])
def predict_heart_condition():
    # Get file from request
    file = request.files["audio_file"]

    # Save the uploaded file to a temporary location
    temp_file_path = "temp_audio_file.wav"
    file.save(temp_file_path)

    try:
        # Call heart_prediction function with the file path
        old_model_pred_class, old_model_confidence = old_model_pred(temp_file_path)
        new_model_pred_class, new_model_confidence = new_model_pred(temp_file_path)
    finally:
        # Delete the temporary audio file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # Convert confidence to float
    new_model_confidence = float(new_model_confidence)
    old_model_confidence = float(old_model_confidence)

    # Return prediction as JSON response
    return jsonify({
            "oldModelPrediction": {
                "prediction": old_model_pred_class,
                "confidence": old_model_confidence
            },
            "newModelPrediction": {
                "prediction": new_model_pred_class,
                "confidence": new_model_confidence
            }
        })

