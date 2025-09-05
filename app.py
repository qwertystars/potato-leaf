from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import joblib
import numpy as np
import pandas as pd
import uuid
import logging
from datetime import datetime
import random

# Configure logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from feature_extractor import extract_features
    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError:
    FEATURE_EXTRACTOR_AVAILABLE = False
    extract_features = None
    print("⚠️ feature_extractor not available")

try:
    from advanced_feature_extractor import predict_with_heatmap, AdvancedFeatureExtractor
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    predict_with_heatmap = None
    AdvancedFeatureExtractor = None
    print("⚠️ advanced_feature_extractor not available")

try:
    from nlp_utils import generate_treatment_summary
    NLP_UTILS_AVAILABLE = True
except ImportError:
    NLP_UTILS_AVAILABLE = False
    generate_treatment_summary = None
    print("⚠️ nlp_utils not available")

try:
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    sns = None
    matplotlib = None
    plt = None
    print("⚠️ plotting libraries not available")

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    import requests
    from weather_service import WeatherService
    from prediction_engine import PredictionEngine
    from weather_database import WeatherDatabase
    WEATHER_AVAILABLE = True
except ImportError:
    WEATHER_AVAILABLE = False
    load_dotenv = None
    requests = None
    WeatherService = None
    PredictionEngine = None
    WeatherDatabase = None
    print("⚠️ weather functionality not available")

# Mobile processing (optional)
try:
    from mobile_image_processor import process_mobile_upload
    MOBILE_PROCESSOR_AVAILABLE = True
except ImportError:
    MOBILE_PROCESSOR_AVAILABLE = False
    process_mobile_upload = None
    print("⚠️ mobile_image_processor not available")

# Camera upload functionality
try:
    from backend.src.api.camera_upload_routes import camera_bp
    CAMERA_UPLOAD_AVAILABLE = True
except ImportError:
    CAMERA_UPLOAD_AVAILABLE = False
    camera_bp = None
    print("⚠️ camera upload functionality not available")

app = Flask(__name__, template_folder="html_template")
app.config['SECRET_KEY'] = 'your-secret-key-change-this'

# Initialize weather services
weather_service = None
prediction_engine = None
weather_db = None
if WEATHER_AVAILABLE:
    try:
        # Initialize weather services (no API key required for Open-Meteo)
        weather_service = WeatherService()
        prediction_engine = PredictionEngine()
        weather_db = WeatherDatabase()
        print("✅ Weather services initialized")
    except Exception as e:
        print(f"⚠️ Weather services initialization failed: {e}")
        WEATHER_AVAILABLE = False

# Initialize chat system
try:
    from chat_integration import init_socketio, setup_chat_routes, init_chat_system, chat_manager
    socketio = init_socketio(app)
    setup_chat_routes(app)
    init_chat_system()
    CHAT_AVAILABLE = True
    print("✅ Chat system integrated")
except ImportError as e:
    socketio = None
    CHAT_AVAILABLE = False
    print(f"⚠️ Chat system not available: {e}")

# Register camera upload blueprint if available
if CAMERA_UPLOAD_AVAILABLE and camera_bp:
    app.register_blueprint(camera_bp)

# Make zip available in templates
app.jinja_env.globals.update(zip=zip)

# Create necessary directories
UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Disease information database
DISEASE_INFO = {
    "healthy": {
        "name": "Healthy Leaf",
        "description": "The leaf appears to be healthy with no signs of disease.",
        "treatment": "No treatment needed. Continue with regular care and monitoring.",
        "prevention": "Maintain good agricultural practices, proper watering, and regular monitoring."
    },
    "early_blight": {
        "name": "Early Blight",
        "description": "Early blight is a common fungal disease that affects potato plants, causing dark spots on leaves.",
        "treatment": "Apply copper-based fungicides or neem oil. Remove affected leaves. Ensure proper spacing for air circulation.",
        "prevention": "Crop rotation, proper spacing, avoid overhead watering, and resistant varieties."
    },
    "late_blight": {
        "name": "Late Blight",
        "description": "Late blight is a serious fungal disease that can destroy entire potato crops rapidly.",
        "treatment": "Apply fungicides immediately. Remove and destroy affected plants. Improve drainage and air circulation.",
        "prevention": "Use resistant varieties, avoid overhead irrigation, and apply preventive fungicides in humid conditions."
    },
    "pest_damage": {
        "name": "Pest Damage",
        "description": "Pest damage can be caused by various insects that feed on potato leaves.",
        "treatment": "Identify the specific pest and apply appropriate insecticide. Use biological controls when possible.",
        "prevention": "Regular monitoring, companion planting, and integrated pest management practices."
    },
    "bacteria": {
        "name": "Bacterial Disease",
        "description": "Bacterial infections can cause various symptoms including leaf spots and wilting.",
        "treatment": "Remove affected plants, apply copper-based bactericides, improve air circulation.",
        "prevention": "Use pathogen-free seeds, avoid overhead watering, and practice crop rotation."
    },
    "fungi": {
        "name": "Fungal Disease",
        "description": "Fungal infections are common in humid conditions and can severely damage crops.",
        "treatment": "Apply appropriate fungicides, improve drainage, remove affected plant material.",
        "prevention": "Ensure proper spacing, avoid overhead watering, and use resistant varieties."
    },
    "virus": {
        "name": "Viral Disease",
        "description": "Viral diseases are typically spread by insects and can cause stunting and discoloration.",
        "treatment": "Remove infected plants, control vector insects, use virus-free planting material.",
        "prevention": "Control insect vectors, use certified disease-free seeds, and practice good sanitation."
    },
    "phytopthora": {
        "name": "Phytophthora",
        "description": "Phytophthora is a serious water mold that causes root rot and leaf blight.",
        "treatment": "Improve drainage, apply specific fungicides, remove affected plants.",
        "prevention": "Ensure good drainage, avoid overwatering, and use resistant varieties."
    }
}

# Available model types
AVAILABLE_MODELS = {
    'svm': 'SVM (Original)',
    'densenet': 'DenseNet-121 (Advanced)',
    'vgg16': 'VGG-16 (Advanced)',
    'resnet': 'ResNet-50 (Advanced)'
}

# Model loading
svm_loaded = False
svm_clf = None
svm_le = None

def load_svm_model():
    """Load SVM model if available"""
    global svm_loaded, svm_clf, svm_le
    try:
        if os.path.exists("models/svm_model_2048.pkl"):
            svm_data = joblib.load("models/svm_model_2048.pkl")
            svm_clf = svm_data["clf"]
            svm_le = svm_data["le"]
            svm_loaded = True
            print("✅ SVM model loaded successfully")
        else:
            print("❌ SVM model file not found")
    except Exception as e:
        print(f"❌ Error loading SVM model: {e}")

# Weather functions
def get_weather_risk_level(humidity, temperature, description):
    """Calculate disease risk based on weather conditions"""
    risk_score = 0

    # Humidity risk
    if humidity >= 85:
        risk_score += 40
    elif humidity >= 70:
        risk_score += 25
    elif humidity >= 60:
        risk_score += 10

    # Temperature risk
    if 20 <= temperature <= 30:
        risk_score += 20
    elif 15 <= temperature <= 35:
        risk_score += 10

    # Weather condition risk
    risky_conditions = ['rain', 'drizzle', 'shower', 'thunderstorm', 'mist', 'fog']
    if any(condition in description.lower() for condition in risky_conditions):
        risk_score += 20

    # Risk level determination
    if risk_score >= 60:
        return "Very High"
    elif risk_score >= 40:
        return "High"
    elif risk_score >= 20:
        return "Medium"
    else:
        return "Low"

def get_simple_weather(city):
    """Get simple weather data for a city using Open-Meteo API via weather service"""
    if not WEATHER_AVAILABLE or not city or city.strip() == "":
        return {
            "city": city or "Not provided",
            "temperature": 25,
            "humidity": 65,
            "description": "Partly cloudy",
            "risk": "Medium",
            "status": "mock_data"
        }

    if not weather_service:
        return {
            "city": city,
            "temperature": 25,
            "humidity": 65,
            "description": "Partly cloudy",
            "risk": "Medium",
            "status": "mock_data",
            "note": "Weather service not available"
        }

    try:
        # Geocode the city to get coordinates
        geocode_result = weather_service.geocode_location(city.strip())
        if not geocode_result:
            raise Exception("Location not found")
        
        lat, lon, formatted_name = geocode_result
        
        # Get current weather (first hour from forecast)
        forecast_result = weather_service.fetch_weather_forecast(lat, lon, formatted_name)
        
        if not forecast_result['success'] or not forecast_result['forecast']:
            raise Exception("Weather data unavailable")
        
        # Use current hour data (first forecast entry)
        current_weather = forecast_result['forecast'][0]
        
        temp = round(current_weather['temperature'], 1)
        humidity = current_weather['humidity']
        desc = current_weather['weather_description']
        risk = get_weather_risk_level(humidity, temp, desc)

        return {
            "city": formatted_name,
            "temperature": temp,
            "humidity": humidity,
            "description": desc.title(),
            "risk": risk,
            "status": "success",
            "country": ""  # Open-Meteo doesn't provide country separately
        }

    except Exception as e:
        logger.error(f"Weather fetch error for {city}: {e}")
        return {
            "city": city,
            "temperature": "-",
            "humidity": "-",
            "description": "Weather unavailable",
            "risk": "Unknown",
            "status": "error"
        }

def get_weather_data(city):
    """Get weather data for a city - alias for get_simple_weather"""
    return get_simple_weather(city)

def predict_disease_svm(image_path):
    """Predict disease using SVM model"""
    if not svm_loaded or not FEATURE_EXTRACTOR_AVAILABLE:
        return None

    try:
        features = extract_features(image_path).reshape(1, -1)
        probabilities = svm_clf.predict_proba(features)[0]
        predicted_class_idx = svm_clf.predict(features)[0]
        predicted_class = svm_le.inverse_transform([predicted_class_idx])[0]

        return {
            "predicted_class": predicted_class,
            "confidence": probabilities[predicted_class_idx],
            "all_probabilities": probabilities,
            "class_names": list(svm_le.classes_)
        }
    except Exception as e:
        print(f"Error in SVM prediction: {e}")
        return None

def create_prediction_chart(result, filename, model_type):
    """Create a probability bar chart for all classes and a pie chart for top 3 classes"""
    if not PLOTTING_AVAILABLE:
        return None, None

    try:
        class_labels = [cls.replace("_", " ").title() for cls in result["class_names"]]
        probabilities_percent = np.array(result["all_probabilities"]) * 100

        # --- Bar Chart (all classes) ---
        df = pd.DataFrame({
            'Disease': class_labels,
            'Probability': probabilities_percent
        })
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(data=df, y='Disease', x='Probability', palette="viridis", orient='h')
        pred_class_label = result["predicted_class"].replace("_", " ").title()
        for i, bar in enumerate(ax.patches):
            if class_labels[i] == pred_class_label:
                bar.set_edgecolor('red')
                bar.set_linewidth(3)
        ax.set_title(f'Disease Classification Results\nModel: {AVAILABLE_MODELS.get(model_type, model_type)}\nPredicted: {pred_class_label}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Probability (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Disease Classes', fontsize=14, fontweight='bold')
        for i, bar in enumerate(ax.patches):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', ha='left', va='center', fontweight='bold', fontsize=12)
        ax.set_xlim(0, max(probabilities_percent) * 1.15)
        sns.despine()
        plt.tight_layout()
        chart_filename = f"chart_{model_type}_{uuid.uuid4().hex[:8]}_{filename}.png"
        chart_path = os.path.join(RESULTS_FOLDER, chart_filename)
        plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

        # --- Pie Chart (top 3 classes) ---
        top3_idx = np.argsort(probabilities_percent)[-3:][::-1]
        top3_labels = [class_labels[i] for i in top3_idx]
        top3_probs = [probabilities_percent[i] for i in top3_idx]
        plt.figure(figsize=(7, 7))
        colors = sns.color_palette("husl", len(top3_labels))
        patches, texts, autotexts = plt.pie(top3_probs, labels=top3_labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize': 13, 'fontweight': 'bold'})
        plt.title(f'Top 3 Predicted Classes', fontsize=15, fontweight='bold')
        plt.axis('equal')
        pie_chart_filename = f"pie_{model_type}_{uuid.uuid4().hex[:8]}_{filename}.png"
        pie_chart_path = os.path.join(RESULTS_FOLDER, pie_chart_filename)
        plt.savefig(pie_chart_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

        return chart_filename, pie_chart_filename
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None, None

# Initialize models
load_svm_model()

@app.route("/")
def index():
    """Main page"""
    # Check available models
    available_models = {}

    if svm_loaded:
        available_models['svm'] = AVAILABLE_MODELS['svm']

    # Check for advanced models
    if ADVANCED_MODELS_AVAILABLE:
        for model_type in ['densenet', 'vgg16', 'resnet']:
            model_path = f"models/{model_type}_leaf_disease_model.h5"
            if os.path.exists(model_path):
                available_models[model_type] = AVAILABLE_MODELS[model_type]

    return render_template("index.html",
                         available_models=available_models,
                         weather_available=WEATHER_AVAILABLE,
                         mobile_processor_available=MOBILE_PROCESSOR_AVAILABLE)

@app.route("/mobile-camera")
def mobile_camera():
    """Mobile camera page"""
    return render_template("mobile_camera.html")

@app.route("/farmer-chat")
def farmer_chat():
    """Farmer chat page"""
    return render_template("farmer_chat.html")

@app.route("/weather-prediction")
def weather_prediction():
    """Weather-based disease prediction page"""
    return render_template("weather_prediction.html")

@app.route("/treatment-guide/<disease>")
def treatment_guide(disease):
    """Comprehensive treatment guide for a specific disease"""
    
    # Normalize disease name
    disease_key = disease.strip().lower().replace(" ", "_").replace("-", "_")
    
    # Treatment data mapping
    treatment_data = {
        "early_blight": {
            "display_name": "Early Blight (Alternaria solani)",
            "icon": "🍂",
            "description": "Early blight is a common fungal disease that affects potato plants, causing dark spots on leaves with concentric rings. It typically appears during warm, humid weather conditions.",
            "chemical_treatments": [
                "Mancozeb 75% WP (0.25% solution)",
                "Chlorothalonil 75% WP (0.2% solution)",
                "Azoxystrobin 23% SC (0.1% solution)",
                "Difenoconazole 25% EC (0.05% solution)",
                "Pyraclostrobin 20% WG",
                "Iprodione 50% WP (0.25% solution)",
                "Fluopyram + Tebuconazole mixture",
                "Trifloxystrobin + Difenoconazole combination"
            ],
            "organic_treatments": [
                "Neem oil spray (2–3% concentration)",
                "Compost tea foliar application",
                "Trichoderma harzianum biofungicide",
                "Fermented cow dung + cow urine spray",
                "Garlic & onion extract foliar spray",
                "Aloe vera leaf extract spray",
                "Diluted buttermilk foliar spray"
            ],
            "preventive_measures": [
                "Crop rotation with cereals or legumes",
                "Mulching to reduce soil splash",
                "Wide row spacing for better airflow",
                "Destroy plant debris after harvest",
                "Use disease-free certified seed tubers",
                "Plant resistant varieties like Kufri Jyoti, Kufri Bahar",
                "Avoid excessive nitrogen fertilizers"
            ],
            "additional_advice": "Apply treatments early in the morning or late evening. Ensure good coverage of leaf surfaces. Rotate between different fungicide groups to prevent resistance."
        },
        "late_blight": {
            "display_name": "Late Blight (Phytophthora infestans)",
            "icon": "❄",
            "description": "Late blight is a serious fungal disease that can destroy entire potato crops rapidly. It thrives in cool, wet conditions and can spread quickly throughout the field.",
            "chemical_treatments": [
                "Metalaxyl 8% + Mancozeb 64% WP",
                "Propineb 70% WP (0.25% solution)",
                "Cymoxanil + Mancozeb (0.3% solution)",
                "Dimethomorph 50% WP (0.2% solution)",
                "Famoxadone + Cymoxanil mixture",
                "Fluopicolide + Propamocarb combination",
                "Oxathiapiprolin (newer molecule)",
                "Mandipropamid 23% SC"
            ],
            "organic_treatments": [
                "Bordeaux mixture (1% solution)",
                "Copper oxychloride spray",
                "Garlic + ginger extract spray",
                "Mustard oil emulsion spray",
                "Neem oil (2% concentration)",
                "Chitosan-based biofungicides"
            ],
            "preventive_measures": [
                "Grow resistant potato varieties",
                "Improve drainage (avoid waterlogging)",
                "Use drip irrigation (avoid sprinkler systems)",
                "Spray preventively during humid weather",
                "Destroy volunteer potato plants near field",
                "Early sowing to escape late blight peak season",
                "Proper plant spacing for ventilation"
            ],
            "additional_advice": "Monitor weather conditions closely. Apply fungicides before disease symptoms appear during favorable conditions. Remove infected plants immediately."
        },
        "healthy": {
            "display_name": "Healthy Plant",
            "icon": "✅",
            "description": "Your potato plant appears to be healthy with no signs of disease. Continue with good agricultural practices to maintain plant health.",
            "chemical_treatments": [
                "No chemical treatment needed",
                "Regular monitoring for early detection of any issues"
            ],
            "organic_treatments": [
                "Continue with organic soil amendments",
                "Compost application for soil health",
                "Beneficial microorganism inoculation"
            ],
            "preventive_measures": [
                "Maintain good agricultural practices",
                "Proper watering schedule",
                "Regular monitoring for pests and diseases",
                "Balanced fertilization",
                "Good field hygiene",
                "Crop rotation planning"
            ],
            "additional_advice": "Keep monitoring your plants regularly. Maintain optimal growing conditions and continue preventive practices to ensure continued plant health."
        }
    }
    
    # Get confidence from query parameter
    confidence = request.args.get('confidence', 'N/A')
    
    # Get treatment info for the disease
    disease_info = treatment_data.get(disease_key)
    
    if not disease_info:
        # Default info for unknown diseases
        disease_info = {
            "display_name": disease.replace("_", " ").title(),
            "icon": "🦠",
            "description": "Disease information not available in our database.",
            "chemical_treatments": ["Consult with agricultural experts for proper treatment"],
            "organic_treatments": ["Consult with agricultural experts for organic alternatives"],
            "preventive_measures": ["Follow general good agricultural practices"],
            "additional_advice": "Contact local agricultural extension services for specific treatment recommendations."
        }
    
    return render_template("treatment_guide.html",
                         disease_name=disease,
                         disease_display_name=disease_info["display_name"],
                         disease_icon=disease_info["icon"],
                         disease_description=disease_info["description"],
                         chemical_treatments=disease_info["chemical_treatments"],
                         organic_treatments=disease_info["organic_treatments"],
                         preventive_measures=disease_info["preventive_measures"],
                         additional_advice=disease_info["additional_advice"],
                         confidence=confidence)

@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint"""
    try:
        # Validate file upload
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template("index.html", error="Please select an image file")

        file = request.files['file']

        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return render_template("index.html", error="Please upload a valid image file")

        # Get form parameters
        model_type = request.form.get("model_type", "svm")
        city = request.form.get("city", "").strip()
        generate_heatmap = request.form.get("generate_heatmap") == "on"
        selected_lang_code = request.form.get("lang", "en")
        mobile_processing_enabled = request.form.get("mobile_processing") == "on"
        
        # Convert language code to language name for backward compatibility
        language_map = {
            'en': 'English',
            'hi': 'Hindi', 
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'bn': 'Bengali',
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'ur': 'Urdu',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'tr': 'Turkish',
            'pl': 'Polish',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish',
            'he': 'Hebrew',
            'fa': 'Persian'
        }
        selected_lang = language_map.get(selected_lang_code, 'English')

        # Save uploaded file with unique name
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        # Optional mobile processing
        processed_filename = None
        validation_info = None
        predict_path = filepath
        if MOBILE_PROCESSOR_AVAILABLE and mobile_processing_enabled:
            try:
                processed_path, validation_info = process_mobile_upload(filepath, save_processed=True)
                if processed_path and os.path.exists(processed_path):
                    processed_filename = os.path.basename(processed_path)
                    predict_path = processed_path
            except Exception as e:
                print(f"Mobile processing failed: {e}")

        # Make prediction based on model type
        if model_type == "svm":
            if not svm_loaded:
                return render_template("index.html", error="SVM model not available")

            result = predict_disease_svm(predict_path)
            if result is None:
                return render_template("index.html", error="Error making SVM prediction")

            heatmap_filename = None

        else:
            # Try advanced model prediction
            if not ADVANCED_MODELS_AVAILABLE:
                return render_template("index.html", error="Advanced models not available")

            result, heatmap_path = predict_with_heatmap(
                predict_path,
                model_type=model_type,
                save_heatmap=generate_heatmap
            )

            if result is None:
                return render_template("index.html", error=f"{model_type} model not available")

            heatmap_filename = os.path.basename(heatmap_path) if heatmap_path else None

        # Get disease information
        disease_key = result["predicted_class"].lower()
        disease_info = DISEASE_INFO.get(disease_key, {
            "name": result["predicted_class"].replace("_", " ").title(),
            "description": "Disease information not available.",
            "treatment": "Consult with agricultural experts for proper treatment.",
            "prevention": "Follow general good agricultural practices."
        })

        # Get weather data
        weather_data = get_weather_data(city) if city else None

        # Generate treatment advice
        treatment_advice = disease_info["treatment"]
        if NLP_UTILS_AVAILABLE:
            try:
                nlp_advice = generate_treatment_summary(result["predicted_class"], selected_lang)
                treatment_advice = f"{disease_info['treatment']}<br><br><strong>AI Generated Advice ({selected_lang}):</strong><br>{nlp_advice}"
            except Exception as _:
                pass

        # Create visualization chart
        chart_filename = None
        pie_chart_filename = None
        if PLOTTING_AVAILABLE:
            chart_filename, pie_chart_filename = create_prediction_chart(result, unique_filename, model_type)

        # Prepare data for template
        predicted_disease = result["predicted_class"].replace("_", " ").title()
        confidence_percent = round(float(result["confidence"]) * 100, 2)

        # Prepare probabilities for template
        classes = [cls.replace("_", " ").title() for cls in result["class_names"]]
        probabilities = [round(float(prob) * 100, 2) for prob in result["all_probabilities"]]

        # Prepare weather data for template
        weather_template_data = None
        if weather_data:
            weather_template_data = {
                'city': weather_data.get('city', 'Unknown'),
                'temp': weather_data.get('temperature', '-'),
                'humidity': weather_data.get('humidity', '-'),
                'desc': weather_data.get('description', 'Unknown'),
                'risk': weather_data.get('risk', 'Unknown'),
                'status': weather_data.get('status', 'unknown'),
                'note': weather_data.get('note', ''),
                'country': weather_data.get('country', '')
            }

        return render_template("result.html",
                             disease=predicted_disease,
                             confidence=confidence_percent,
                             classes=classes,
                             probability=probabilities,
                             filename=unique_filename,
                             processed_filename=processed_filename,
                             validation_info=validation_info,
                             bar_chart=chart_filename,
                             pie_chart=pie_chart_filename,
                             heatmap_path=heatmap_filename,
                             heatmap_available=heatmap_filename is not None,
                             weather=weather_template_data,
                             advice=treatment_advice,
                             model_used=AVAILABLE_MODELS.get(model_type, model_type))

    except Exception as e:
        print(f"Error in prediction: {e}")
        return render_template("index.html", error=f"An error occurred: {str(e)}")

@app.route("/get_disease_info/<disease_name>")
def get_disease_info(disease_name):
    """Get detailed information about a specific disease"""
    # Normalize disease name: replace spaces and hyphens with underscores, lowercase
    disease_key = disease_name.strip().lower().replace(" ", "_").replace("-", "_")
    info = DISEASE_INFO.get(disease_key, {})

    if not info:
        return jsonify({"error": "Disease information not found"})

    return jsonify(info)

@app.route("/check_models")
def check_models():
    """Check which models are available"""
    available_models = {}

    if svm_loaded:
        available_models['svm'] = {
            "name": AVAILABLE_MODELS['svm'],
            "status": "loaded"
        }

    if ADVANCED_MODELS_AVAILABLE:
        for model_type in ['densenet', 'vgg16', 'resnet']:
            model_path = f"models/{model_type}_leaf_disease_model.h5"
            if os.path.exists(model_path):
                available_models[model_type] = {
                    "name": AVAILABLE_MODELS[model_type],
                    "status": "available"
                }

    return jsonify({
        "available_models": available_models,
        "features": {
            "weather": WEATHER_AVAILABLE,
            "plotting": PLOTTING_AVAILABLE,
            "nlp": NLP_UTILS_AVAILABLE,
            "advanced_models": ADVANCED_MODELS_AVAILABLE,
            "mobile_processing": MOBILE_PROCESSOR_AVAILABLE
        }
    })

@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/static/results/<filename>")
def result_file(filename):
    """Serve result files"""
    return send_from_directory(RESULTS_FOLDER, filename)

# New: Serve dataset images safely
@app.route("/dataset-image/<category>/<path:filename>")
def dataset_image(category, filename):
    """Serve images from the labeled datasets (early_blight, late_blight, healthy)."""
    allowed = {
        "early_blight": os.path.join("datasets", "early_blight"),
        "late_blight": os.path.join("datasets", "late_blight"),
        "healthy": os.path.join("datasets", "healthy"),
    }
    base_dir = allowed.get(category)
    if not base_dir:
        return jsonify({"error": "Invalid category"}), 400
    # Prevent path traversal: send_from_directory handles safe joins
    if not os.path.exists(base_dir):
        return jsonify({"error": "Category not available"}), 404
    return send_from_directory(base_dir, filename)

# New: Quiz page
@app.route("/quiz")
def disease_quiz():
    """Interactive quiz page that uses labeled dataset images."""
    return render_template("disease_quiz.html")

# New: API to get random quiz images
@app.route("/api/quiz/random-images", methods=["GET"])
def api_quiz_random_images():
    """Return a random selection of labeled images from datasets.
    Query params: count (int, 1-50)
    """
    try:
        count = request.args.get("count", default=5, type=int)
        if count is None or count <= 0:
            count = 5
        count = max(1, min(50, count))

        categories = {
            "Early Blight": ("early_blight", os.path.join("datasets", "early_blight")),
            "Late Blight": ("late_blight", os.path.join("datasets", "late_blight")),
            "Healthy": ("healthy", os.path.join("datasets", "healthy")),
        }

        # Build a pool of available (label, category, filename)
        pool = []
        for label, (cat_key, dir_path) in categories.items():
            if os.path.isdir(dir_path):
                try:
                    files = [f for f in os.listdir(dir_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"))]
                except Exception:
                    files = []
                for f in files:
                    pool.append({
                        "label": label,
                        "category": cat_key,
                        "filename": f,
                        "url": f"/dataset-image/{cat_key}/{f}"
                    })
        if not pool:
            return jsonify({"success": False, "error": "No dataset images available"}), 404

        # Sample without replacement; if count > len(pool), cap to len(pool)
        k = min(count, len(pool))
        selected = random.sample(pool, k)

        # Assign ids and hide raw filename in primary payload (still included in url path)
        for i, item in enumerate(selected):
            item["id"] = i + 1
        return jsonify({
            "success": True,
            "count": len(selected),
            "items": selected
        })
    except Exception as e:
        logger.error(f"Quiz random images error: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

# Chat API Routes
@app.route("/api/chat/rooms", methods=["GET"])
def api_get_chat_rooms():
    """Get list of chat rooms"""
    try:
        from backend.src.database import get_session
        from backend.src.services.chat_service import ChatService
        
        db = get_session()
        try:
            chat_service = ChatService(db)
            rooms = chat_service.get_all_rooms()
            return jsonify({
                "rooms": [
                    {
                        "id": room.id,
                        "name": room.name,
                        "description": room.description,
                        "topic": room.topic,
                        "created_at": room.created_at.isoformat(),
                        "active_users": len(chat_manager.get_room_users(room.id))
                    }
                    for room in rooms
                ]
            })
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error getting chat rooms: {e}")
        return jsonify({"error": "Failed to get chat rooms"}), 500

@app.route("/api/chat/rooms/<room_id>/messages", methods=["GET"])
def api_get_room_messages(room_id):
    """Get messages for a specific room"""
    try:
        from backend.src.database import get_session
        from backend.src.services.chat_service import ChatService
        
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        db = get_session()
        try:
            chat_service = ChatService(db)
            messages = chat_service.get_room_messages(room_id, limit, offset)
            return jsonify({
                "messages": [
                    {
                        "id": msg.id,
                        "user_id": msg.user_id,
                        "content": msg.content,
                        "timestamp": msg.sent_at.isoformat(),
                        "sent_at": msg.sent_at.isoformat()
                    }
                    for msg in messages
                ]
            })
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error getting room messages: {e}")
        return jsonify({"error": "Failed to get messages"}), 500

@app.route("/api/chat/rooms/<room_id>/messages", methods=["POST"])
def api_send_room_message(room_id):
    """Send a message to a room"""
    try:
        from backend.src.database import get_session
        from backend.src.services.chat_service import ChatService
        
        data = request.json
        user_id = data.get('user_id')
        content = data.get('content')
        
        if not user_id or not content:
            return jsonify({"error": "user_id and content are required"}), 400
        
        db = get_session()
        try:
            chat_service = ChatService(db)
            message = chat_service.create_message(room_id, user_id, content)
            return jsonify({
                "id": message.id,
                "user_id": message.user_id,
                "content": message.content,
                "timestamp": message.sent_at.isoformat()
            })
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return jsonify({"error": "Failed to send message"}), 500

@app.route("/api/chat/users/<user_id>/profile", methods=["POST"])
def api_create_user_profile(user_id):
    """Create or update user profile"""
    try:
        from backend.src.database import get_session
        from backend.src.services.chat_service import ChatService
        
        data = request.json or {}
        display_name = data.get('display_name', f"Farmer_{user_id[-4:]}")
        
        db = get_session()
        try:
            chat_service = ChatService(db)
            # Try to get existing profile first
            profile = chat_service.get_user_profile(user_id)
            if not profile:
                profile = chat_service.create_user_profile(user_id, display_name)
            
            return jsonify({
                "user_id": profile.user_id,
                "display_name": profile.display_name,
                "created_at": profile.created_at.isoformat(),
                "last_active": profile.last_active.isoformat() if profile.last_active else None
            })
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error creating user profile: {e}")
        return jsonify({"error": "Failed to create user profile"}), 500

# Language API Routes
@app.route("/api/languages")
def api_get_languages():
    """Get available languages"""
    try:
        from backend.src.database import SessionLocal
        from backend.src.services.language_service import LanguageService
        
        db = SessionLocal()
        try:
            language_service = LanguageService(db)
            languages = language_service.get_available_languages()
            default_language = language_service.get_default_language()
            
            return jsonify({
                "languages": languages,
                "defaultLanguage": default_language
            })
        finally:
            db.close()
    except Exception as e:
        print(f"Error getting languages: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/user/language-preference", methods=["GET"])
def api_get_user_language_preference():
    """Get user's current language preference"""
    try:
        from backend.src.database import SessionLocal
        from backend.src.services.language_service import LanguageService
        
        session_id = request.headers.get('session-id')
        user_id = request.headers.get('user-id')
        
        db = SessionLocal()
        try:
            language_service = LanguageService(db)
            preference = language_service.get_user_language_preference(user_id, session_id)
            
            if not preference:
                # Return default language if no preference set
                default_language = language_service.get_default_language()
                return jsonify({
                    "languageCode": default_language,
                    "preferenceSource": "default",
                    "selectedAt": None,
                    "isActive": True
                })
            
            return jsonify(preference)
        finally:
            db.close()
    except Exception as e:
        print(f"Error getting user language preference: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/user/language-preference", methods=["POST"])
def api_set_user_language_preference():
    """Set user's language preference"""
    try:
        from backend.src.database import SessionLocal
        from backend.src.services.language_service import LanguageService
        from backend.src.models.language import PreferenceSource
        
        data = request.json
        language_code = data.get('language_code')
        source = data.get('source', 'user_selection')
        
        session_id = request.headers.get('session-id')
        user_id = request.headers.get('user-id')
        
        if not language_code:
            return jsonify({"error": "language_code is required"}), 400
        
        db = SessionLocal()
        try:
            language_service = LanguageService(db)
            
            # Convert string to enum
            preference_source = PreferenceSource.USER_SELECTION
            if source:
                preference_source = PreferenceSource(source)
            
            preference = language_service.set_user_language_preference(
                language_code,
                user_id,
                session_id,
                preference_source
            )
            
            # Start translation session for analytics
            language_service.start_translation_session(session_id, language_code)
            
            return jsonify(preference)
        except ValueError as e:
            print(f"Invalid language preference request: {e}")
            return jsonify({"error": str(e)}), 400
        finally:
            db.close()
    except Exception as e:
        print(f"Error setting user language preference: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/translations/<language_code>")
def api_get_translations(language_code):
    """Get translations for a specific language"""
    try:
        from backend.src.database import SessionLocal
        from backend.src.services.language_service import LanguageService
        
        namespace = request.args.get('namespace')
        
        db = SessionLocal()
        try:
            language_service = LanguageService(db)
            translations = language_service.get_translations(language_code, namespace)
            
            return jsonify({
                "languageCode": language_code,
                "namespace": namespace,
                "translations": translations
            })
        finally:
            db.close()
    except Exception as e:
        print(f"Error getting translations: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/ai-translate", methods=["POST"])
def api_translate_content():
    """Translate AI-generated content"""
    try:
        from backend.src.database import SessionLocal
        from backend.src.services.language_service import LanguageService, TranslationService
        
        data = request.json
        content = data.get('content')
        target_language = data.get('target_language')
        source_language = data.get('source_language', 'en')
        context = data.get('context', 'general')
        use_cache = data.get('use_cache', True)
        
        session_id = request.headers.get('session-id')
        
        if not content or not target_language:
            return jsonify({"error": "content and target_language are required"}), 400
        
        db = SessionLocal()
        try:
            translation_service = TranslationService(db)
            language_service = LanguageService(db)
            
            # Update session activity for analytics
            if session_id:
                language_service.update_session_activity(session_id, ai_translation=True)
            
            result = translation_service.translate_content(
                content,
                target_language,
                source_language,
                context,
                use_cache
            )
            
            # Add translation time for response
            result["translationTime"] = 45  # Mock value
            
            return jsonify(result)
        finally:
            db.close()
    except Exception as e:
        print(f"Error translating content: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/translation-session/activity", methods=["POST"])
def api_update_session_activity():
    """Update session activity for analytics"""
    try:
        from backend.src.database import SessionLocal
        from backend.src.services.language_service import LanguageService
        
        data = request.json or {}
        page_view = data.get('page_view', False)
        ai_translation = data.get('ai_translation', False)
        
        session_id = request.headers.get('session-id')
        
        if not session_id:
            return jsonify({"error": "session-id header is required"}), 400
        
        with SessionLocal() as db:
            language_service = LanguageService(db)
            language_service.update_session_activity(
                session_id, page_view, ai_translation
            )
            
        return jsonify({"success": True})
        
    except Exception as e:
        print(f"Error updating session activity: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Google Translate Session API Routes
@app.route("/api/language/session/persist", methods=["POST"])
def api_persist_language_session():
    """Persist language preference in session for Google Translate integration"""
    try:
        from flask import session
        
        data = request.json
        language = data.get('language')
        session_id = data.get('session_id') or request.headers.get('session-id')
        
        if not language:
            return jsonify({"error": "language is required"}), 400
        
        # Store in Flask session
        session['language_preference'] = language
        session['language_session_id'] = session_id
        session['language_updated_at'] = datetime.now().isoformat()
        
        # Also store in backend database if available
        try:
            from backend.src.database import SessionLocal
            from backend.src.services.language_service import LanguageService
            
            db = SessionLocal()
            try:
                language_service = LanguageService(db)
                language_service.set_user_language_preference(
                    language,
                    user_id=request.headers.get('user-id'),
                    session_id=session_id,
                    source='google_translate'
                )
            finally:
                db.close()
        except Exception as db_error:
            logger.warning(f"Could not save to database: {db_error}")
        
        return jsonify({
            "success": True,
            "language": language,
            "session_id": session_id,
            "timestamp": session.get('language_updated_at')
        })
        
    except Exception as e:
        logger.error(f"Error persisting language session: {e}")
        return jsonify({"error": "Failed to persist language preference"}), 500

@app.route("/api/language/session/current", methods=["GET"])
def api_get_current_language_session():
    """Get current language preference from session"""
    try:
        from flask import session
        
        session_id = request.headers.get('session-id')
        
        # Check Flask session first
        language = session.get('language_preference')
        updated_at = session.get('language_updated_at')
        
        # Fallback to database if session is empty
        if not language:
            try:
                from backend.src.database import SessionLocal
                from backend.src.services.language_service import LanguageService
                
                db = SessionLocal()
                try:
                    language_service = LanguageService(db)
                    preference = language_service.get_user_language_preference(
                        user_id=request.headers.get('user-id'),
                        session_id=session_id
                    )
                    if preference:
                        language = preference.get('languageCode', 'en')
                        updated_at = preference.get('selectedAt')
                finally:
                    db.close()
            except Exception as db_error:
                logger.warning(f"Could not load from database: {db_error}")
        
        # Final fallback to default
        if not language:
            language = 'en'
        
        return jsonify({
            "language": language,
            "session_id": session_id,
            "updated_at": updated_at,
            "source": "session" if session.get('language_preference') else "default"
        })
        
    except Exception as e:
        logger.error(f"Error getting current language session: {e}")
        return jsonify({"error": "Failed to get language preference"}), 500

@app.route("/api/language/session/update", methods=["PUT"])
def api_update_language_session():
    """Update language preference in session"""
    try:
        from flask import session
        
        data = request.json
        language = data.get('language')
        session_id = data.get('session_id') or request.headers.get('session-id')
        
        if not language:
            return jsonify({"error": "language is required"}), 400
        
        # Update Flask session
        old_language = session.get('language_preference')
        session['language_preference'] = language
        session['language_session_id'] = session_id
        session['language_updated_at'] = datetime.now().isoformat()
        
        # Update database if available
        try:
            from backend.src.database import SessionLocal
            from backend.src.services.language_service import LanguageService
            
            db = SessionLocal()
            try:
                language_service = LanguageService(db)
                language_service.set_user_language_preference(
                    language,
                    user_id=request.headers.get('user-id'),
                    session_id=session_id,
                    source='google_translate_update'
                )
            finally:
                db.close()
        except Exception as db_error:
            logger.warning(f"Could not update database: {db_error}")
        
        return jsonify({
            "success": True,
            "language": language,
            "previous_language": old_language,
            "session_id": session_id,
            "timestamp": session.get('language_updated_at')
        })
        
    except Exception as e:
        logger.error(f"Error updating language session: {e}")
        return jsonify({"error": "Failed to update language preference"}), 500

# Weather API Routes
@app.route("/api/weather/forecast", methods=["GET"])
def api_weather_forecast():
    """Get 3-day weather forecast for location"""
    if not WEATHER_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Weather functionality not available"
        }), 503
    
    if not weather_service:
        return jsonify({
            "success": False,
            "error": "Weather service not available"
        }), 503
    
    try:
        # Get parameters
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        location_name = request.args.get('location_name', '')
        
        # Validate parameters
        if lat is None or lon is None:
            return jsonify({
                "success": False,
                "error": "Latitude and longitude are required"
            }), 400
        
        if not (-90 <= lat <= 90):
            return jsonify({
                "success": False,
                "error": "Invalid latitude. Must be between -90 and 90"
            }), 400
        
        if not (-180 <= lon <= 180):
            return jsonify({
                "success": False,
                "error": "Invalid longitude. Must be between -180 and 180"
            }), 400
        
        # Fetch weather forecast
        result = weather_service.fetch_weather_forecast(lat, lon, location_name)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 503
            
    except Exception as e:
        logger.error(f"Weather forecast API error: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route("/api/weather/geocode", methods=["GET"])
def api_weather_geocode():
    """Convert location name to coordinates"""
    if not WEATHER_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Weather functionality not available"
        }), 503
    
    if not weather_service:
        return jsonify({
            "success": False,
            "error": "Weather service not available"
        }), 503
    
    try:
        location = request.args.get('location', '').strip()
        
        if not location:
            return jsonify({
                "success": False,
                "error": "Location parameter is required"
            }), 400
        
        if len(location) > 200:
            return jsonify({
                "success": False,
                "error": "Location name too long (max 200 characters)"
            }), 400
        
        # Geocode location
        result = weather_service.geocode_location(location)
        
        if result:
            lat, lon, formatted_name = result
            return jsonify({
                "success": True,
                "location": {
                    "lat": lat,
                    "lon": lon,
                    "name": formatted_name,
                    "query": location
                }
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": f"Location '{location}' not found"
            }), 404
            
    except Exception as e:
        logger.error(f"Geocoding API error: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route("/api/prediction/disease-spread", methods=["POST"])
def api_disease_spread_prediction():
    """Calculate disease spread risk based on weather data"""
    logger.info("Disease spread prediction endpoint called")
    
    if not WEATHER_AVAILABLE or not prediction_engine:
        logger.warning(f"Service unavailable - WEATHER_AVAILABLE: {WEATHER_AVAILABLE}, prediction_engine: {prediction_engine is not None}")
        return jsonify({
            "success": False,
            "error": "Prediction service not available"
        }), 503
    
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")
        
        if not data:
            logger.warning("No JSON data received")
            return jsonify({
                "success": False,
                "error": "JSON data required"
            }), 400
        
        # Validate required fields
        required_fields = ['disease_type', 'lat', 'lon', 'weather_data']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            logger.warning(f"Missing required fields: {missing_fields}")
            return jsonify({
                "success": False,
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        disease_type = data['disease_type']
        lat = data['lat']
        lon = data['lon']
        weather_data = data['weather_data']
        
        logger.info(f"Processing request for disease: {disease_type}, location: ({lat}, {lon}), weather_data length: {len(weather_data) if isinstance(weather_data, list) else 'not a list'}")
        
        # Validate disease type
        valid_diseases = ['Early_Blight', 'Late_Blight', 'Healthy']
        if disease_type not in valid_diseases:
            logger.warning(f"Invalid disease type: {disease_type}")
            return jsonify({
                "success": False,
                "error": f"Invalid disease type. Must be one of: {', '.join(valid_diseases)}"
            }), 400
        
        # Validate coordinates
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            logger.warning(f"Invalid coordinates: lat={lat}, lon={lon}")
            return jsonify({
                "success": False,
                "error": "Invalid coordinates"
            }), 400
        
        # Validate weather data
        if not isinstance(weather_data, list) or not weather_data:
            logger.warning(f"Invalid weather data: type={type(weather_data)}, length={len(weather_data) if hasattr(weather_data, '__len__') else 'N/A'}")
            return jsonify({
                "success": False,
                "error": "Weather data must be a non-empty list"
            }), 400
        
        # Log sample weather data for debugging
        logger.info(f"Sample weather data (first item): {weather_data[0] if weather_data else 'None'}")
        
        # Calculate disease spread prediction
        logger.info("Calling prediction_engine.predict_disease_spread")
        prediction_result = prediction_engine.predict_disease_spread(
            weather_data, disease_type, lat, lon
        )
        logger.info(f"Prediction completed successfully: {prediction_result}")
        
        return jsonify({
            "success": True,
            "prediction": prediction_result
        }), 200
        
    except Exception as e:
        logger.error(f"Disease prediction API error: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route("/api/prediction/visualize", methods=["GET"])
def api_prediction_visualize():
    """Get visualization data for weather and disease prediction"""
    if not WEATHER_AVAILABLE or not weather_db:
        return jsonify({
            "success": False,
            "error": "Visualization service not available"
        }), 503
    
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        if lat is None or lon is None:
            return jsonify({
                "success": False,
                "error": "Latitude and longitude are required"
            }), 400
        
        # Get weather forecast data
        weather_forecasts = weather_db.get_weather_forecast_by_location(lat, lon)
        
        # Get prediction history
        prediction_history = weather_db.get_prediction_history(lat, lon, days=7)
        
        # Format data for Chart.js
        visualization_data = {
            "weather_trends": {
                "dates": [],
                "temperature": [],
                "humidity": [],
                "precipitation": []
            },
            "risk_history": {
                "dates": [],
                "risk_scores": [],
                "risk_levels": []
            },
            "current_conditions": {}
        }
        
        # Process weather data
        if weather_forecasts:
            # Group by date and calculate daily averages
            daily_weather = {}
            for forecast in weather_forecasts:
                date_key = forecast['forecast_date']
                if date_key not in daily_weather:
                    daily_weather[date_key] = {
                        'temperatures': [],
                        'humidities': [],
                        'precipitations': []
                    }
                
                daily_weather[date_key]['temperatures'].append(forecast['temperature'])
                daily_weather[date_key]['humidities'].append(forecast['humidity'])
                daily_weather[date_key]['precipitations'].append(forecast['precipitation'])
            
            # Calculate daily averages
            for date_key in sorted(daily_weather.keys()):
                day_data = daily_weather[date_key]
                visualization_data["weather_trends"]["dates"].append(date_key)
                visualization_data["weather_trends"]["temperature"].append(
                    round(sum(day_data['temperatures']) / len(day_data['temperatures']), 1)
                )
                visualization_data["weather_trends"]["humidity"].append(
                    round(sum(day_data['humidities']) / len(day_data['humidities']), 1)
                )
                visualization_data["weather_trends"]["precipitation"].append(
                    round(sum(day_data['precipitations']), 1)
                )
        
        # Process prediction history
        for prediction in prediction_history:
            visualization_data["risk_history"]["dates"].append(prediction['prediction_date'])
            visualization_data["risk_history"]["risk_scores"].append(prediction['risk_score'])
            visualization_data["risk_history"]["risk_levels"].append(prediction['risk_level'])
        
        # Add current conditions if available
        if weather_forecasts:
            latest_forecast = weather_forecasts[0]
            visualization_data["current_conditions"] = {
                "temperature": latest_forecast['temperature'],
                "humidity": latest_forecast['humidity'],
                "precipitation": latest_forecast['precipitation'],
                "weather_condition": latest_forecast['weather_condition'],
                "weather_description": latest_forecast['weather_description']
            }
        
        return jsonify({
            "success": True,
            "data": visualization_data
        }), 200
        
    except Exception as e:
        logger.error(f"Visualization API error: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return render_template("index.html"), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == "__main__":
    print("🍃 Potato Disease Analyzer")
    print(f"📊 SVM Model: {'✅ Loaded' if svm_loaded else '❌ Not available'}")
    print(f"🔬 Advanced Models: {'✅ Available' if ADVANCED_MODELS_AVAILABLE else '❌ Not available'}")
    print(f"🌤️ Weather Data: {'✅ Available' if WEATHER_AVAILABLE else '❌ Not available'}")
    print(f"📈 Plotting: {'✅ Available' if PLOTTING_AVAILABLE else '❌ Not available'}")
    print(f"📱 Mobile Processing: {'✅ Available' if MOBILE_PROCESSOR_AVAILABLE else '❌ Not available'}")
    print(f"💬 Chat System: {'✅ Available' if CHAT_AVAILABLE else '❌ Not available'}")
    
    if socketio:
        print("🚀 Starting with SocketIO support...")
        socketio.run(app, debug=True, host="0.0.0.0", port=5000)
    else:
        print("🚀 Starting without chat functionality...")
        app.run(debug=True, host="0.0.0.0", port=5000)
