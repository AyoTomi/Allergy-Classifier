from suggestion_engine import (
    generate_safe_ingredients,
    suggest_suppressive_measures,
    get_allergen_category_map_from_dataset,
    generate_fact_based_suggestions,
    load_scientific_knowledge
)
from allergen_utils import load_allergen_category_map
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
from tensorflow.keras.preprocessing import image

# === PATHS ===
MODEL_PATH = "C:\\Users\\TOMISIN\\Documents\\food allergy detection\\main\\food_classifier_model.h5"
LABELS = sorted(os.listdir("c:/Users/TOMISIN/Documents/food allergy detection/firsttry/data/nigerian_food_dataset/images/train"))
INGREDIENT_JSON_PATH = "C:/Users/TOMISIN/Documents/food allergy detection/dataset_with_ingredients_prob.json"
HEALTH_CSV_PATH = "c:/Users/TOMISIN/Documents/food allergy detection/firsttry/data/cleaned_data.csv"
SCIENTIFIC_KB_PATH = "C:/Users/TOMISIN/Documents/food allergy detection/scientific_knowledge.json"

# === Load Category Map & Knowledge Base ===
ALLERGEN_CATEGORY_MAP = get_allergen_category_map_from_dataset(INGREDIENT_JSON_PATH)
knowledge_base = load_scientific_knowledge(SCIENTIFIC_KB_PATH)

# === SETTINGS ===
IMG_SIZE = (224, 224)

# === Load Model ===
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

# === Load JSON and Health CSV ===
with open(INGREDIENT_JSON_PATH, 'r') as f:
    ingredient_data = json.load(f)

health_df = pd.read_csv(HEALTH_CSV_PATH)
health_df.columns = health_df.columns.str.strip().str.lower()

# === Prediction Function ===
def predict_food(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence = float(prediction[class_index])
    predicted_food = LABELS[class_index]

    return predicted_food, confidence

# === Get Ingredients & Allergens ===
def get_ingredients_and_allergens(food_name):
    for entry in ingredient_data:
        if entry["food_name"].lower() == food_name.lower():
            return entry["ingredients"], entry["allergens"]
    return [], []

# === Check Allergies ===
def check_user_allergy(user_id, allergens):
    row = health_df[health_df['user_id'].astype(str) == str(user_id)]

    if row.empty:
        return f"❌ User ID {user_id} not found.", None

    allergy_category = str(row.iloc[0]['food_type']).strip().lower()
    is_allergic = int(row.iloc[0]['allergic'])

    food_allergen_classes = {
        ALLERGEN_CATEGORY_MAP.get(a.lower(), "unknown") for a in allergens
    }

    if is_allergic:
        if allergy_category in food_allergen_classes:
            matched = [a for a in allergens if ALLERGEN_CATEGORY_MAP.get(a.lower()) == allergy_category]
            return (
                f"⚠️ User is allergic to {allergy_category}, and this food contains: {', '.join(matched)}",
                allergy_category
            )
        else:
            return (
                f"⚠️ User has an allergy to {allergy_category}, but this food contains different allergen classes: {', '.join(food_allergen_classes)}",
                allergy_category
            )
    else:
        return "✅ No known allergies for this user.", None

# === Suggest Alternative Meals ===
def suggest_alternative_meals(user_allergy_class, ingredient_data, allergen_class_map):
    alternatives = []
    for item in ingredient_data:
        food_allergens = item.get("allergens", [])
        food_allergen_classes = {
            allergen_class_map.get(a.lower(), "unknown") for a in food_allergens
        }
        if user_allergy_class not in food_allergen_classes:
            alternatives.append(item.get("food_name"))
    return list(dict.fromkeys(alternatives))[:5]

# === Display Allergen Info from KB ===
def show_allergen_info(allergen, ingredients):
    allergen_key = allergen.lower()
    if allergen_key in knowledge_base:
        info = knowledge_base[allergen_key]

        # Symptoms
        symptoms = info.get("common_symptoms", [])
        if symptoms:
            print(f"\n🩺 Common Symptoms of {allergen.capitalize()} Allergy:")
            print(f"   - {', '.join(symptoms)}")

        # Suppressive measures
        suppressive_actions = suggest_suppressive_measures([allergen], knowledge_base)
        if suppressive_actions and allergen_key in suppressive_actions:
            print(f"\n🛡️ Suppressive Measures for {allergen.capitalize()}:")
            print(f"   - {', '.join(suppressive_actions[allergen_key])}")

        # Ingredient replacements
        replacements = generate_safe_ingredients(ingredients, [allergen], knowledge_base)
        if replacements and allergen_key in replacements:
            print(f"\n🍽️ Suggested Ingredient Replacements for {allergen.capitalize()}:")
            print(f"   - {', '.join(replacements[allergen_key])}")

        # Justification
        justification = info.get("justification", "")
        if justification:
            print(f"\n📖 Why? {justification}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    sample_image_path = "c:/Users/TOMISIN/Documents/food allergy detection/main/download (1).jpg"
    sample_user_id = "2"

    food, confidence = predict_food(sample_image_path)
    print(f"\nPredicted Food: {food} ({confidence*100:.2f}% confidence)")

    ingredients, allergens = get_ingredients_and_allergens(food)
    print(f"Ingredients: {', '.join(ingredients)}")
    print(f"Allergens: {', '.join(allergens)}")

    result, allergy_category = check_user_allergy(sample_user_id, allergens)
    print(f"👤 Allergy Check for User {sample_user_id}: {result}")

    if "⚠️" in result and allergy_category != "other":
        allergen_in_food = any(
            ALLERGEN_CATEGORY_MAP.get(a.lower(), "unknown") == allergy_category
            for a in allergens
        )

        if allergen_in_food:
            # Show only info for matching allergens
            for allergen in allergens:
                if ALLERGEN_CATEGORY_MAP.get(allergen.lower(), "unknown") == allergy_category:
                    show_allergen_info(allergen, ingredients)

        else:
            # Ask for safe meal suggestions
            choice_alt = input(f"\nYour allergen ({allergy_category}) is not in this meal. Show safe meal suggestions? (yes/no): ").strip().lower()
            if choice_alt == "yes":
                suggestions = suggest_alternative_meals(allergy_category, ingredient_data, ALLERGEN_CATEGORY_MAP)
                if suggestions:
                    print("🍽️ Recommended Meals:", ", ".join(suggestions))
                else:
                    print("⚠️ No safe alternative meals found.")
            else:
                print("✅ Skipping alternative meal suggestions.")

            # Ask if they want to know about other allergens
            choice_other = input("\nDo you want to know about other allergens present in this meal? (yes/no): ").strip().lower()
            if choice_other == "yes":
                for allergen in allergens:
                    if ALLERGEN_CATEGORY_MAP.get(allergen.lower(), "unknown") != allergy_category:
                        show_allergen_info(allergen, ingredients)
            else:
                print("ℹ️ Skipping other allergen details.")

    else:
        print("\n✅ No allergy detected — no suppressive measures or replacements needed.")
