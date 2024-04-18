from flask import Flask, render_template, request

import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath = 'model.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")


def pred_leaf(leaf):
    test_image = load_img(leaf, target_size=(128, 128))  # load image
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255  # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # change dimention 3D to 4D

    result = model.predict(test_image)  # predict palnt
    print('@@ Raw result = ', result)
    pred = np.argmax(result, axis=1)
    print(pred)
    if pred == 0:
        return "Alstonia Scholaris", 'Alstonia_Scholaris.html'

    elif pred == 1:
        return "Arjun", 'Arjun.html'

    elif pred == 2:
        return "Basil", 'Basil.html'

    elif pred == 3:
        return "Chinar", 'Chinar.html'

    elif pred == 4:
        return "Guava", 'Guava.html'

    elif pred == 5:
        return "Jamun", 'Jamun.html'

    elif pred == 6:
        return "Jatropha", 'Jatropha.html'

    elif pred == 7:
        return "Lemon", 'Lemon.html'
    elif pred == 8:
        return "Mango", 'Mango.html'

    elif pred == 9:
        return "Pomegranate", 'Pomegranate.html'


# Create flask instance
app = Flask(__name__)


# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')


# get input image from user then predict class and render respective .html page for information
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # fet input
        filename = file.filename
        print("@@ Input posted = ", filename)

        file_path = os.path.join('static/upload', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_leaf(leaf=file_path)

        return render_template(output_page, pred_output=pred, user_image=file_path)


# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False, port=8080)
