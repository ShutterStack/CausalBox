# main.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Add the 'routers' and 'utils' directories to the Python path
# This allows direct imports like 'from routers.preprocess_routes import preprocess_bp'
script_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(script_dir, 'routers'))
sys.path.insert(0, os.path.join(script_dir, 'utils'))

# Import Blueprints
from routers.preprocess_routes import preprocess_bp
from routers.discover_routes import discover_bp
from routers.intervene_routes import intervene_bp
from routers.treatment_routes import treatment_bp
from routers.visualize_routes import visualize_bp
from routers.prediction_routes import prediction_bp
from routers.timeseries_routes import timeseries_bp
from routers.chatbot_routes import chatbot_bp

app = Flask(__name__)
CORS(app) # Enable CORS for frontend interaction

# Register Blueprints
app.register_blueprint(preprocess_bp, url_prefix='/preprocess')
app.register_blueprint(discover_bp, url_prefix='/discover')
app.register_blueprint(intervene_bp, url_prefix='/intervene')
app.register_blueprint(treatment_bp, url_prefix='/treatment')
app.register_blueprint(visualize_bp, url_prefix='/visualize')
app.register_blueprint(prediction_bp, url_prefix='/prediction')
app.register_blueprint(timeseries_bp, url_prefix='/timeseries')
app.register_blueprint(chatbot_bp, url_prefix='/chatbot')

@app.route('/')
def home():
    return "Welcome to CausalBox Backend API!"

if __name__ == '__main__':
    # Ensure the 'data' directory exists for storing datasets
    os.makedirs('data', exist_ok=True)
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)