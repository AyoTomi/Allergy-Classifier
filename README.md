# AI Food Allergy Classifier

An intelligent food recognition and allergy detection system that uses deep learning to identify Nigerian foods and check for potential allergens based on user profiles.

## 🏗️ Architecture

This application uses a **Flask API + Web Frontend** architecture that directly utilizes the original Keras model without conversion:

- **Backend**: Python Flask API that loads the original `.h5` Keras model
- **Frontend**: Modern HTML5 web interface with real-time AI predictions
- **AI Model**: Deep CNN trained on 27 Nigerian food categories
- **Smart Fallback**: Automatic demo mode when API is unavailable

## 📁 Project Structure

```
Allergy-Classifier/
├── app.py                      # Flask API server (main backend)
├── allergy-classifier.html     # Web frontend interface
├── food_classifier_model.h5    # Trained Keras model
├── dataset_with_ingredients_prob.json  # Food ingredients & allergens data
├── cleaned_data.csv            # User health/allergy profiles
├── scientific_knowledge.json   # Scientific knowledge base
├── requirements.txt            # Python dependencies
├── start_api.bat              # Quick start script for Windows
├── convert_model.py           # Optional: TensorFlow.js converter
└── README.md                  # This file
```

## 🚀 Quick Start

### Method 1: One-Click Start (Windows)
1. **Double-click** `start_api.bat`
2. Wait for "Server starting on http://localhost:5000"
3. **Open** `allergy-classifier.html` in your browser
4. **Upload** a food image and analyze!

### Method 2: Manual Setup
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start the Flask API server
python app.py

# 3. Open the web interface
# Open allergy-classifier.html in your browser
```

## 📋 Requirements

### Python Dependencies:
- Python 3.8+
- Flask 2.3.3
- TensorFlow 2.13.0
- OpenCV 4.8.0
- Pandas 2.0.3
- NumPy 1.24.3
- Pillow 10.0.0

### Data Files Required:
- `food_classifier_model.h5` - Your trained Keras model
- `dataset_with_ingredients_prob.json` - Food ingredients database
- `cleaned_data.csv` - User allergy profiles
- `scientific_knowledge.json` - Knowledge base (optional)

## 🎯 Features

### Core Functionality:
- **🤖 AI Food Recognition**: 27 Nigerian food categories with confidence scores
- **⚠️ Allergy Detection**: Real-time allergen checking against user profiles  
- **📊 Smart Analytics**: Detailed ingredient analysis and recommendations
- **🎨 Modern UI**: Professional interface with drag-and-drop uploads
- **📱 Mobile Friendly**: Responsive design for all devices

### Advanced Features:
- **🔄 Auto-Fallback**: Works offline with simulation mode
- **⚡ Real-time Processing**: Fast predictions using original Keras model
- **🎭 Visual Feedback**: Color-coded warnings and animated confidence bars
- **🔍 Health Integration**: Cross-references with user medical profiles
- **💡 Smart Recommendations**: Alternative meal suggestions for allergic users

## 🎮 How to Use

### Step 1: Start the System
Run `start_api.bat` or `python app.py`

### Step 2: Open Web Interface
Open `allergy-classifier.html` in any modern browser

### Step 3: Configure User
- Enter Patient ID (1-4 for demo users)
- Demo profiles include various allergy types

### Step 4: Analyze Food
- **Upload**: Click or drag-and-drop food image
- **Analyze**: Click "Analyze Food & Check Allergies"
- **Review**: Check results for allergen warnings

## 👥 Demo User Profiles

The system includes sample user data for testing:

| User ID | Allergy Type | Status |
|---------|-------------|---------|
| 1 | Nuts | Allergic |
| 2 | Dairy | Allergic |
| 3 | Shellfish | Allergic |
| 4 | None | No Allergies |

## 🍽️ Supported Foods (27 Categories)

Nigerian cuisine classification including:
- **Rice dishes**: Jollof rice, Fried rice
- **Soups**: Egusi, Okra, Pepper soup, Bitter leaf
- **Proteins**: Suya, Fish stew, Chicken stew
- **Sides**: Dodo, Akara, Moi moi
- **Starches**: Fufu, Pounded yam, Eba, Amala
- And many more traditional dishes

## 🔧 API Endpoints

The Flask backend provides these REST endpoints:

- `GET /` - API documentation
- `GET /health` - System status check
- `POST /predict` - Food classification and allergy analysis
- `GET /ingredients/<food_name>` - Get food ingredients
- `GET /user/<user_id>/allergies` - Get user allergy profile

## 🎨 Interface Modes

### 🤖 Keras Model Active
- Real AI predictions using your trained model
- Shows "Keras Model Active" indicator
- Full accuracy and performance

### 🎭 Demo Mode  
- Simulation when API unavailable
- Shows "Demo Mode" indicator
- Educational purposes and offline testing

## 🛠️ Troubleshooting

### Common Issues:

**Problem**: "API not available" message
```bash
Solution: 
1. Check if Python is installed
2. Run: pip install -r requirements.txt
3. Start API: python app.py
4. Refresh browser page
```

**Problem**: Model not found error
```bash
Solution:
1. Ensure food_classifier_model.h5 is in the project folder
2. Check file path in app.py line 18
3. Verify model file isn't corrupted
```

**Problem**: Browser shows CORS error
```bash
Solution:
1. API includes flask-cors for cross-origin requests
2. Try different browser or disable security temporarily
3. Ensure API is running on localhost:5000
```

## 🔮 Advanced Configuration

### Custom Model Path:
Edit `app.py` line 18 to change model location:
```python
MODEL_PATH = "path/to/your/model.h5"
```

### Add New Food Categories:
Update the `LABELS` array in both `app.py` and `allergy-classifier.html`

### Custom User Data:
Replace `cleaned_data.csv` with your user profiles following the same format

## 🚀 Production Deployment

For production use:
1. Use a production WSGI server (gunicorn, uWSGI)
2. Set up proper database instead of CSV files
3. Implement user authentication
4. Add HTTPS and security headers
5. Optimize model loading for better performance

## 📈 Performance Notes

- **Model Loading**: ~2-3 seconds on first startup
- **Prediction Time**: ~0.5-1 second per image
- **Image Processing**: Supports JPG, PNG, GIF up to 10MB
- **Memory Usage**: ~500MB with loaded model

## 🤝 Contributing

To contribute or modify:
1. Fork the repository
2. Make changes to `app.py` for backend logic
3. Update `allergy-classifier.html` for UI changes
4. Test both API and simulation modes
5. Submit pull request with clear description

## 📄 License

This project is for educational and research purposes. Please ensure you have proper licensing for any commercial use of the trained model and datasets.
