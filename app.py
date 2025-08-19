from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
import pandas as pd
import os
from tensorflow.keras.preprocessing import image
import base64
from io import BytesIO
from PIL import Image
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# === PATHS ===
MODEL_PATH = "food_classifier_model.h5"
INGREDIENT_JSON_PATH = "dataset_with_ingredients_prob.json"
HEALTH_CSV_PATH = "cleaned_data.csv"
SCIENTIFIC_KB_PATH = "scientific_knowledge.json"

# === SETTINGS ===
IMG_SIZE = (224, 224)

# === Load Model ===
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# === Load Labels ===
LABELS = [
    "Abacha and Ugba(african salad)", "Akara", "Amala", "Banga soup",
    "Beans and Plantain", "Bitter leaf soup", "Chicken stew",
    "Dodo (fried plantain)", "Eba", "Egusi soup", "Fish stew",
    "Fried rice", "Fufu", "Gbegiri soup", "Jollof rice", "Moi moi",
    "Nkwobi", "Nsala soup", "Ofe nsala (white soup)", "Ogbono soup",
    "Okra soup", "Pepper soup", "Pounded yam", "Suya",
    "Tuwo shinkafa", "Vegetable soup", "Yam porridge"
]

# === Load Data ===
ingredient_data = []
health_df = None
knowledge_base = {}

try:
    with open(INGREDIENT_JSON_PATH, 'r') as f:
        ingredient_data = json.load(f)
    print(f"✅ Loaded {len(ingredient_data)} ingredient records")
except Exception as e:
    print(f"❌ Error loading ingredient data: {e}")

try:
    health_df = pd.read_csv(HEALTH_CSV_PATH)
    health_df.columns = health_df.columns.str.strip().str.lower()
    print(f"✅ Loaded {len(health_df)} health records")
except Exception as e:
    print(f"❌ Error loading health data: {e}")

try:
    with open(SCIENTIFIC_KB_PATH, 'r') as f:
        knowledge_base = json.load(f)
    print(f"✅ Loaded knowledge base with {len(knowledge_base)} entries")
except Exception as e:
    print(f"❌ Error loading knowledge base: {e}")

# === Allergen Category Map ===
ALLERGEN_CATEGORY_MAP = {
    "legumes": "legumes",
    "crayfish": "shellfish", 
    "stockfish": "fish",
    "nuts/seeds": "nuts",
    "dairy": "dairy",
    "gluten": "gluten",
    "soy": "soy",
    "eggs": "eggs"
}

def preprocess_image(img_data):
    """Preprocess image for model prediction"""
    try:
        # Convert base64 to PIL Image
        img_data = img_data.split(',')[1]  # Remove data:image/jpeg;base64,
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array and resize
        img_array = np.array(img)
        img_array = cv2.resize(img_array, IMG_SIZE)
        
        # Normalize pixel values
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def get_ingredients_and_allergens(food_name):
    """Get ingredients and allergens for a food"""
    for entry in ingredient_data:
        if entry["food_name"].lower() == food_name.lower():
            return entry.get("ingredients", []), entry.get("allergens", [])
    return [], []

def check_user_allergy(user_id, allergens):
    """Check if user has allergies to the food"""
    if health_df is None:
        return {
            "message": "❌ Health data not available",
            "isAllergic": False,
            "allergyCategory": None
        }
    
    row = health_df[health_df['user_id'].astype(str) == str(user_id)]
    
    if row.empty:
        return {
            "message": f"❌ User ID {user_id} not found.",
            "isAllergic": False,
            "allergyCategory": None
        }
    
    allergy_category = str(row.iloc[0]['food_type']).strip().lower()
    is_allergic = int(row.iloc[0]['allergic'])
    
    if not is_allergic:
        return {
            "message": "✅ No known allergies for this user.",
            "isAllergic": False,
            "allergyCategory": None
        }
    
    # Check if any allergens match user's allergy category
    food_allergen_classes = {
        ALLERGEN_CATEGORY_MAP.get(a.lower(), "unknown") for a in allergens
    }
    
    if allergy_category in food_allergen_classes:
        matched = [a for a in allergens if ALLERGEN_CATEGORY_MAP.get(a.lower()) == allergy_category]
        return {
            "message": f"⚠️ User is allergic to {allergy_category}, and this food contains: {', '.join(matched)}",
            "isAllergic": True,
            "allergyCategory": allergy_category,
            "matchedAllergens": matched
        }
    else:
        return {
            "message": f"⚠️ User has an allergy to {allergy_category}, but this food contains different allergen classes: {', '.join(food_allergen_classes)}",
            "isAllergic": True,
            "allergyCategory": allergy_category
        }

@app.route('/')
def home():
    return """
    <h1>🍽️ Food Allergy Classifier API</h1>
    <p>Backend API for food classification and allergy detection</p>
    <p>Endpoints:</p>
    <ul>
        <li><strong>POST /predict</strong> - Predict food from image</li>
        <li><strong>GET /health</strong> - Check API status</li>
    </ul>
    """

@app.route('/health')
def health_check():
    """API health check"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "ingredients_loaded": len(ingredient_data) > 0,
        "health_data_loaded": health_df is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        user_id = data.get('userId', '1')
        
        # Preprocess image
        img_array = preprocess_image(data['image'])
        if img_array is None:
            return jsonify({"error": "Error processing image"}), 400
        
        # Make prediction
        prediction = model.predict(img_array)[0]
        class_index = np.argmax(prediction)
        confidence = float(prediction[class_index])
        predicted_food = LABELS[class_index]
        
        # Get ingredients and allergens
        ingredients, allergens = get_ingredients_and_allergens(predicted_food)
        
        # Check allergies
        allergy_check = check_user_allergy(user_id, allergens)
        
        return jsonify({
            "prediction": {
                "food": predicted_food,
                "confidence": confidence,
                "classIndex": int(class_index)
            },
            "ingredients": ingredients,
            "allergens": allergens,
            "allergyCheck": allergy_check,
            "success": True
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/ingredients/<food_name>')
def get_food_info(food_name):
    """Get ingredients and allergens for a specific food"""
    ingredients, allergens = get_ingredients_and_allergens(food_name)
    return jsonify({
        "food": food_name,
        "ingredients": ingredients,
        "allergens": allergens
    })

@app.route('/user/<user_id>/allergies')
def get_user_allergies(user_id):
    """Get user allergy information"""
    if health_df is None:
        return jsonify({"error": "Health data not available"}), 500
    
    row = health_df[health_df['user_id'].astype(str) == str(user_id)]
    
    if row.empty:
        return jsonify({"error": f"User {user_id} not found"}), 404
    
    return jsonify({
        "userId": user_id,
        "allergyType": str(row.iloc[0]['food_type']).strip(),
        "isAllergic": int(row.iloc[0]['allergic']) == 1
    })

if __name__ == '__main__':
    print("🚀 Starting Food Allergy Classifier API...")
    print(f"📊 Model loaded: {'✅' if model else '❌'}")
    print(f"📋 Ingredient data: {len(ingredient_data)} items")
    print(f"🏥 Health data: {len(health_df) if health_df is not None else 0} users")
    print("🌐 Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
