from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import torch.nn as nn
import joblib
import os
import logging
from datetime import datetime
import traceback
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost"])  # Restrict CORS for security; adjust as needed

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class URLClassifierAPI:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.model_info = {'status': 'not_loaded', 'error': None}
        self.load_model()
    
    def load_model(self):
        """Load the trained model and components"""
        try:
            logger.info("Loading URL classification model...")
            
            # Check if model files exist
            required_files = [
                'tfidf_vectorizer.joblib',
                'label_encoder.joblib',
                'url_classifier_pytorch_model.pth'
            ]
            
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if missing_files:
                error_msg = f"Missing model files: {', '.join(missing_files)}"
                logger.error(error_msg)
                self.model_info = {
                    'status': 'error',
                    'error': error_msg,
                    'missing_files': missing_files
                }
                return
            
            # Load individual components
            self.load_individual_components()
                
            if self.model is not None:
                logger.info("✅ Model loaded successfully!")
                logger.info(f"Model info: {self.model_info}")
            else:
                logger.error("❌ Failed to load model")
                
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.model_info = {
                'status': 'error',
                'error': error_msg
            }
    
    def load_individual_components(self):
        """Load individual model components"""
        # Load vectorizer
        self.vectorizer = joblib.load('tfidf_vectorizer.joblib')
        logger.info("✅ Loaded vectorizer")
        
        # Load label encoder
        self.label_encoder = joblib.load('label_encoder.joblib')
        logger.info("✅ Loaded label encoder")
        
        # Load model
        input_dim = len(self.vectorizer.get_feature_names_out())
        num_classes = len(self.label_encoder.classes_)
        
        self.model = LogisticRegressionModel(input_dim, num_classes)
        self.model.load_state_dict(torch.load('url_classifier_pytorch_model.pth', map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        logger.info("✅ Loaded PyTorch model")
        
        # Post-load dimension check
        if self.model.linear.in_features != input_dim:
            raise ValueError(f"Model input dimension mismatch: expected {input_dim}, got {self.model.linear.in_features}")
        
        self.model_info = {
            'status': 'loaded',
            'input_dim': input_dim,
            'num_classes': num_classes,
            'accuracy': 'N/A',
            'classes': self.label_encoder.classes_.tolist(),
            'loaded_from': 'individual_components'
        }
    
    def predict_url(self, url):
        """Predict if a URL is malicious or benign"""
        try:
            if self.model is None:
                return {
                    'error': 'Model not loaded',
                    'details': self.model_info.get('error', 'Unknown error')
                }
            
            # Validate URL
            try:
                parsed = urlparse(url)
                if not all([parsed.scheme, parsed.netloc]):
                    return {'error': 'Invalid URL format'}
            except Exception:
                return {'error': 'Invalid URL'}
            
            # Preprocess the URL (assuming model was trained on raw strings; adjust if needed)
            url_tfidf = self.vectorizer.transform([str(url)])
            url_tensor = torch.FloatTensor(url_tfidf.toarray()).to(self.device)
            
            # Make prediction with OOM handling
            try:
                with torch.no_grad():
                    outputs = self.model(url_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(outputs, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
            except torch.cuda.OutOfMemoryError:
                return {'error': 'Out of memory during prediction; try CPU or smaller input'}
            
            # Decode prediction
            predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Create probability dictionary
            prob_dict = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                prob_dict[str(class_name)] = float(probabilities[0][i].item())
            
            # Flexible malicious check (handles string labels like 'malicious' or numeric '1')
            is_malicious = ('malicious' in str(predicted_label).lower()) or (str(predicted_label).lower() == '1')
            
            return {
                'url': url,
                'prediction': str(predicted_label),
                'confidence': float(confidence),
                'probabilities': prob_dict,
                'is_malicious': is_malicious,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting URL {url}: {e}\n{traceback.format_exc()}")
            return {'error': f'Prediction failed: {str(e)}'}

# Initialize the classifier
classifier = URLClassifierAPI()

# HTML template for web interface (simplified for user-facing app)
WEB_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>URL Safety Checker</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container { 
            max-width: 600px; 
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }
        h1 {
            color: #333;
        }
        .error-box {
            background: #ff6b6b;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .result { 
            margin: 20px 0; 
            padding: 15px; 
            border-radius: 5px;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .safe { 
            background-color: #d4edda; 
            border: 1px solid #c3e6cb; 
        }
        .malicious { 
            background-color: #f8d7da; 
            border: 1px solid #f5c6cb; 
        }
        input[type="text"] { 
            width: 80%; 
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            margin-bottom: 10px;
        }
        button { 
            padding: 12px 24px; 
            background: #667eea; 
            color: white; 
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background: #764ba2;
        }
        p.instructions {
            color: #666;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 URL Safety Checker</h1>
        <p class="instructions">Enter a URL to check if it's safe or potentially malicious.</p>
        
        {% if model_info.status == 'error' %}
        <div class="error-box">
            <h3>⚠️ System Error</h3>
            <p>Unable to load the detection model. Please try again later.</p>
        </div>
        {% else %}
        
        <form method="post">
            <input type="text" name="url" placeholder="https://example.com" required>
            <button type="submit">Check URL</button>
        </form>
        
        {% if result %}
            {% if result.error %}
            <div class="error-box">
                <h3>Error</h3>
                <p>{{ result.error }}</p>
            </div>
            {% else %}
            <div class="result {{ 'malicious' if result.is_malicious else 'safe' }}">
                <h3>{{ result.url }}</h3>
                <p><strong>Status:</strong> {{ "🚨 Potentially Malicious" if result.is_malicious else "✅ Safe" }}</p>
                <p><strong>Confidence:</strong> {{ "%.2f"|format(result.confidence * 100) }}%</p>
            </div>
            {% endif %}
        {% endif %}
        
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    """Web interface for testing"""
    return render_template_string(WEB_TEMPLATE, model_info=classifier.model_info)

@app.route('/', methods=['POST'])
def web_predict():
    """Handle web form submission"""
    url = request.form.get('url')
    result = classifier.predict_url(url) if url else None
    return render_template_string(WEB_TEMPLATE, result=result, model_info=classifier.model_info)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for URL prediction"""
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400
        
        url = data['url']
        result = classifier.predict_url(url)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"API error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if classifier.model is not None else 'unhealthy',
        'model_loaded': classifier.model is not None,
        'model_info': classifier.model_info,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/batch', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'urls' not in data:
            return jsonify({'error': 'URLs array is required'}), 400
        
        urls = data['urls']
        if not isinstance(urls, list):
            return jupytext({'error': 'URLs must be an array'}), 400
        
        if len(urls) > 100:
            return jsonify({'error': 'Batch size exceeds limit (max 100)'}), 400
        
        results = []
        for url in urls:
            result = classifier.predict_url(url)
            results.append(result)
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Batch API error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting URL Classifier API...")
    logger.info(f"Model loaded: {'✅' if classifier.model is not None else '❌'}")
    
    if classifier.model_info.get('status') == 'error':
        logger.warning(f"⚠️  Error: {classifier.model_info.get('error')}")
        if classifier.model_info.get('missing_files'):
            logger.warning(f"Missing files: {', '.join(classifier.model_info['missing_files'])}")
            logger.info("\n💡 Run the create_mock_models.py script to generate test model files!")
    
    logger.info("\nAvailable endpoints:")
    logger.info("  - GET  /                    (Web interface)")
    logger.info("  - POST /api/predict         (JSON: {'url': 'http://example.com'})")
    logger.info("  - POST /api/batch          (JSON: {'urls': [...]})")
    logger.info("  - GET  /api/health         (Health check)")
    logger.info("\n🌐 Access at: http://localhost:5000")
    
    # Run with threading disabled to avoid loading issues
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=False)
