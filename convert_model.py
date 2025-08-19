import tensorflowjs as tfjs
import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('food_classifier_model.h5')

# Convert to TensorFlow.js format
tfjs.converters.save_keras_model(model, 'tfjs_model')

print("Model converted to TensorFlow.js format!")
print("Model saved in 'tfjs_model' directory")
