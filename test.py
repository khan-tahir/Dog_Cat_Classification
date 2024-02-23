import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model('cat_dog_classifier.h5')

# Load the image
img_path = 'test_img4.jpg'
img = load_img(img_path, target_size= (64,64))

img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0
prediction = model.predict(img_array)

plt.imshow(img)
plt.axis('off')

class_label = 'Cat' if prediction[0][0] > 0.5 else 'Dog'
plt.title(f'Predicted: {class_label}')

# Show 
plt.show()
