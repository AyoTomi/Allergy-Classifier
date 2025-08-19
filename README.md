# Food Allergy Classifier - TensorFlow.js Implementation

This project converts the Python-based food allergy detection system into a TensorFlow.js implementation that can run in browsers or Node.js environments.

## Files Created

### 1. `allergy-classifier.html`
A complete web-based implementation that runs in the browser:
- **Features**: Image upload (drag & drop), food classification, allergy checking
- **UI**: Clean, responsive design with visual warnings for allergens
- **Dependencies**: Only TensorFlow.js CDN (no local installation needed)
- **Usage**: Open directly in any modern web browser

### 2. `allergy-classifier.js`
A Node.js module implementation:
- **Features**: Complete API for food classification and allergy checking
- **Usage**: Can be imported into other Node.js applications
- **Dependencies**: `@tensorflow/tfjs-node` package

### 3. `convert_model.py`
Python script to convert the Keras model (.h5) to TensorFlow.js format:
- **Purpose**: Converts `food_classifier_model.h5` to browser-compatible format
- **Output**: Creates a `tfjs_model/` directory with model files

## Original Python Implementation Analysis

The original system includes:
- **Model**: Keras CNN for Nigerian food classification (27 food types)
- **Data**: Ingredient and allergen information for each food
- **Health Records**: User allergy profiles
- **Features**: 
  - Food image classification
  - Ingredient identification
  - Allergen detection
  - Alternative meal suggestions
  - Suppressive measures recommendations

## TensorFlow.js Conversion Features

### Core Functionality Preserved:
1. **Food Classification**: Image-based food recognition
2. **Allergen Detection**: Identifies allergens in predicted foods
3. **User Allergy Checking**: Cross-references with user health profiles
4. **Alternative Suggestions**: Recommends safe meal alternatives
5. **Confidence Scoring**: Shows prediction confidence levels

### Enhancements Added:
- **Web Interface**: Drag-and-drop image upload
- **Real-time Analysis**: Instant results without server calls
- **Visual Feedback**: Color-coded allergy warnings
- **Responsive Design**: Works on desktop and mobile devices

## How to Use

### Web Version (Recommended):
1. Open `allergy-classifier.html` in a web browser
2. Enter a User ID (1-4 for demo data)
3. Upload or drag-and-drop a food image
4. Click "Analyze Food" to get results

### Node.js Version:
```bash
npm install @tensorflow/tfjs-node
node allergy-classifier.js
```

### To Use with Real Model:
1. Install TensorFlow.js converter:
   ```bash
   pip install tensorflowjs
   ```

2. Convert the Keras model:
   ```bash
   tensorflowjs_converter --input_format=keras food_classifier_model.h5 tfjs_model
   ```

3. Update the model path in the code:
   ```javascript
   await classifier.loadModel('./tfjs_model/model.json');
   ```

## Sample Data Included

The implementation includes sample data for demonstration:

### Food Categories (27 Nigerian Foods):
- Jollof rice, Egusi soup, Suya, Akara, etc.

### User Health Records:
- User 1: Nuts allergy
- User 2: Dairy allergy  
- User 3: Shellfish allergy
- User 4: No allergies

### Allergen Categories:
- Legumes, Nuts/Seeds, Shellfish, Fish, Dairy, Gluten

## Technical Notes

- **Model Input**: 224x224 RGB images
- **Preprocessing**: Automatic image resizing and normalization
- **Output**: Food classification with confidence scores
- **Memory Management**: Automatic tensor disposal to prevent memory leaks
- **Error Handling**: Comprehensive error catching and user feedback

## Browser Compatibility

- Modern browsers with WebGL support
- Chrome, Firefox, Safari, Edge (latest versions)
- Mobile browsers supported

## Next Steps

1. **Model Conversion**: Convert the actual .h5 model file
2. **Data Integration**: Load complete ingredient and health databases
3. **API Integration**: Connect to backend databases for user data
4. **Performance**: Optimize model size for faster loading
5. **Features**: Add camera capture, batch processing, nutrition info
