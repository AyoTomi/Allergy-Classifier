/**
 * Food Allergy Classifier - TensorFlow.js Implementation
 * Converted from the original Python implementation
 */

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

class FoodAllergyClassifier {
    constructor() {
        this.model = null;
        this.labels = [
            "Abacha and Ugba(african salad)", "Akara", "Amala", "Banga soup",
            "Beans and Plantain", "Bitter leaf soup", "Chicken stew",
            "Dodo (fried plantain)", "Eba", "Egusi soup", "Fish stew",
            "Fried rice", "Fufu", "Gbegiri soup", "Jollof rice", "Moi moi",
            "Nkwobi", "Nsala soup", "Ofe nsala (white soup)", "Ogbono soup",
            "Okra soup", "Pepper soup", "Pounded yam", "Suya",
            "Tuwo shinkafa", "Vegetable soup", "Yam porridge"
        ];
        
        this.ingredientData = [];
        this.healthData = [];
        this.allergenCategoryMap = {
            "legumes": "legumes",
            "crayfish": "shellfish",
            "stockfish": "fish", 
            "nuts/seeds": "nuts",
            "dairy": "dairy",
            "gluten": "gluten"
        };
        
        this.imgSize = [224, 224];
    }

    /**
     * Load the TensorFlow.js model
     * @param {string} modelPath - Path to the model.json file
     */
    async loadModel(modelPath) {
        try {
            console.log('Loading model...');
            this.model = await tf.loadLayersModel(`file://${modelPath}`);
            console.log('Model loaded successfully!');
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            return false;
        }
    }

    /**
     * Load ingredient and allergen data from JSON
     * @param {string} jsonPath - Path to the dataset JSON file
     */
    loadIngredientData(jsonPath) {
        try {
            const data = fs.readFileSync(jsonPath, 'utf8');
            this.ingredientData = JSON.parse(data);
            console.log(`Loaded ${this.ingredientData.length} food items`);
        } catch (error) {
            console.error('Error loading ingredient data:', error);
        }
    }

    /**
     * Load health data from CSV (simplified as JSON for this example)
     * @param {Array} healthData - Array of user health records
     */
    loadHealthData(healthData) {
        this.healthData = healthData;
        console.log(`Loaded ${this.healthData.length} user health records`);
    }

    /**
     * Preprocess image for prediction
     * @param {tf.Tensor} imageBuffer - Image tensor
     * @returns {tf.Tensor} Preprocessed image tensor
     */
    preprocessImage(imageBuffer) {
        // Resize image to model input size
        const resized = tf.image.resizeBilinear(imageBuffer, this.imgSize);
        
        // Normalize pixel values to 0-1
        const normalized = resized.div(255.0);
        
        // Add batch dimension
        const batched = normalized.expandDims(0);
        
        return batched;
    }

    /**
     * Predict food from image
     * @param {string} imagePath - Path to the image file
     * @returns {Object} Prediction result with food name and confidence
     */
    async predictFood(imagePath) {
        if (!this.model) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }

        try {
            // Load and decode image
            const imageBuffer = fs.readFileSync(imagePath);
            const imageTensor = tf.node.decodeImage(imageBuffer, 3);
            
            // Preprocess image
            const preprocessed = this.preprocessImage(imageTensor);
            
            // Make prediction
            const prediction = this.model.predict(preprocessed);
            const probabilities = await prediction.data();
            
            // Find the class with highest probability
            const maxIndex = probabilities.indexOf(Math.max(...probabilities));
            const confidence = probabilities[maxIndex];
            const predictedFood = this.labels[maxIndex];
            
            // Clean up tensors
            imageTensor.dispose();
            preprocessed.dispose();
            prediction.dispose();
            
            return {
                food: predictedFood,
                confidence: confidence
            };
        } catch (error) {
            console.error('Error during prediction:', error);
            throw error;
        }
    }

    /**
     * Get ingredients and allergens for a specific food
     * @param {string} foodName - Name of the food
     * @returns {Object} Object containing ingredients and allergens arrays
     */
    getIngredientsAndAllergens(foodName) {
        const entry = this.ingredientData.find(item => 
            item.food_name.toLowerCase() === foodName.toLowerCase()
        );
        
        return entry ? {
            ingredients: entry.ingredients || [],
            allergens: entry.allergens || []
        } : {
            ingredients: [],
            allergens: []
        };
    }

    /**
     * Check if user has allergies to the food
     * @param {string} userId - User ID to check
     * @param {Array} allergens - Array of allergens in the food
     * @returns {Object} Allergy check result
     */
    checkUserAllergy(userId, allergens) {
        const userRecord = this.healthData.find(user => user.user_id === userId);
        
        if (!userRecord) {
            return {
                message: `❌ User ID ${userId} not found.`,
                isAllergic: false,
                allergyCategory: null
            };
        }

        const allergyCategory = userRecord.food_type.toLowerCase();
        const isAllergic = userRecord.allergic === 1;

        if (!isAllergic) {
            return {
                message: "✅ No known allergies for this user.",
                isAllergic: false,
                allergyCategory: null
            };
        }

        // Map food allergens to categories
        const foodAllergenClasses = new Set();
        allergens.forEach(allergen => {
            const category = this.allergenCategoryMap[allergen.toLowerCase()];
            if (category) {
                foodAllergenClasses.add(category);
            }
        });

        if (foodAllergenClasses.has(allergyCategory)) {
            const matchedAllergens = allergens.filter(allergen => 
                this.allergenCategoryMap[allergen.toLowerCase()] === allergyCategory
            );
            
            return {
                message: `⚠️ User is allergic to ${allergyCategory}, and this food contains: ${matchedAllergens.join(', ')}`,
                isAllergic: true,
                allergyCategory: allergyCategory,
                matchedAllergens: matchedAllergens
            };
        } else {
            return {
                message: `⚠️ User has an allergy to ${allergyCategory}, but this food contains different allergen classes: ${Array.from(foodAllergenClasses).join(', ')}`,
                isAllergic: true,
                allergyCategory: allergyCategory
            };
        }
    }

    /**
     * Suggest alternative meals for users with allergies
     * @param {string} userAllergyClass - User's allergy category
     * @returns {Array} Array of safe food alternatives
     */
    suggestAlternativeMeals(userAllergyClass) {
        const alternatives = [];
        
        for (const item of this.ingredientData) {
            const foodAllergens = item.allergens || [];
            const foodAllergenClasses = new Set();
            
            foodAllergens.forEach(allergen => {
                const category = this.allergenCategoryMap[allergen.toLowerCase()];
                if (category) {
                    foodAllergenClasses.add(category);
                }
            });
            
            if (!foodAllergenClasses.has(userAllergyClass)) {
                alternatives.push(item.food_name);
            }
        }
        
        // Remove duplicates and return first 5
        return [...new Set(alternatives)].slice(0, 5);
    }

    /**
     * Complete food analysis pipeline
     * @param {string} imagePath - Path to the food image
     * @param {string} userId - User ID for allergy checking
     * @returns {Object} Complete analysis results
     */
    async analyzeFoodImage(imagePath, userId) {
        try {
            // Predict food
            const prediction = await this.predictFood(imagePath);
            console.log(`\n🍽️ Predicted Food: ${prediction.food} (${(prediction.confidence * 100).toFixed(2)}% confidence)`);
            
            // Get ingredients and allergens
            const { ingredients, allergens } = this.getIngredientsAndAllergens(prediction.food);
            console.log(`📝 Ingredients: ${ingredients.join(', ')}`);
            console.log(`⚠️ Allergens: ${allergens.join(', ')}`);
            
            // Check allergies
            const allergyCheck = this.checkUserAllergy(userId, allergens);
            console.log(`👤 Allergy Check for User ${userId}: ${allergyCheck.message}`);
            
            // Suggest alternatives if needed
            let alternatives = [];
            if (allergyCheck.isAllergic && allergyCheck.allergyCategory) {
                alternatives = this.suggestAlternativeMeals(allergyCheck.allergyCategory);
                if (alternatives.length > 0) {
                    console.log(`🍽️ Recommended Safe Alternatives: ${alternatives.join(', ')}`);
                }
            }
            
            return {
                prediction: prediction,
                ingredients: ingredients,
                allergens: allergens,
                allergyCheck: allergyCheck,
                alternatives: alternatives
            };
            
        } catch (error) {
            console.error('Error during analysis:', error);
            throw error;
        }
    }
}

// Example usage
async function main() {
    const classifier = new FoodAllergyClassifier();
    
    // Sample health data
    const sampleHealthData = [
        { user_id: "1", food_type: "nuts", allergic: 1 },
        { user_id: "2", food_type: "dairy", allergic: 1 },
        { user_id: "3", food_type: "shellfish", allergic: 1 },
        { user_id: "4", food_type: "gluten", allergic: 0 }
    ];
    
    // Load data
    classifier.loadHealthData(sampleHealthData);
    
    // Load ingredient data (you would load from the actual JSON file)
    const sampleIngredientData = [
        {
            "food_name": "Jollof rice",
            "ingredients": ["rice", "tomatoes", "onions", "pepper", "chicken stock"],
            "allergens": []
        },
        {
            "food_name": "Suya",
            "ingredients": ["beef", "groundnuts", "spices"],
            "allergens": ["nuts/seeds"]
        }
    ];
    classifier.ingredientData = sampleIngredientData;
    
    // Load model (replace with actual model path)
    // const modelLoaded = await classifier.loadModel('./tfjs_model/model.json');
    
    // For demonstration without actual model
    console.log('Food Allergy Classifier initialized successfully!');
    console.log('To use with actual model:');
    console.log('1. Convert your .h5 model using tensorflowjs_converter');
    console.log('2. Load the model using classifier.loadModel(modelPath)');
    console.log('3. Analyze images using classifier.analyzeFoodImage(imagePath, userId)');
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FoodAllergyClassifier;
} else if (typeof window !== 'undefined') {
    window.FoodAllergyClassifier = FoodAllergyClassifier;
}

// Run main function if this file is executed directly
if (require.main === module) {
    main().catch(console.error);
}
