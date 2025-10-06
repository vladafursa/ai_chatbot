import os
from PIL import Image
from tensorflow.keras.models import load_model

classNames = {
    0: 'Cheetah',
    1: 'Lion',
    2: 'Tiger'
}
model = load_model("/Users/vladafursa/AI_submission_T0337485/models/vgg16_classifier.h5")
def getImagePath():
    filePath = input("Enter the path to the image file: ").strip()
    if os.path.isfile(filePath):
        return filePath
    else:
        print("Invalid file path. Please try again.")
import numpy as np
def preprocessImage(imagePath, targetSize=(224, 224)):  # target size
    try:
        image = Image.open(imagePath).convert("RGB")  #  RGB format
        image = image.resize(targetSize)  # resizing
        imageArray = np.array(image) / 255.0  # normalisation
        imageArray = np.expand_dims(imageArray, axis=0)  # add dimension
        return imageArray
    except (AttributeError, FileNotFoundError):
        print("Image is invalid or not found.")
        return None
def classify(imagePath):
    imageArray = preprocessImage(imagePath)
    if imageArray is None:
        return "none of classes"
    else:
        predictions = model.predict(imageArray)  # retrieve predictions
        predictedClass = np.argmax(predictions)  # retrieve the highest probability class
        predictedClassname = classNames[predictedClass]
        return predictedClassname