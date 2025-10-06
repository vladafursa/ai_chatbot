from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import label_binarize

# importing dataset of images
path = 'dataset'

# creating datagen
datagen = ImageDataGenerator(
    rescale=1. / 255,  # scaling images to the [0, 1] range
    validation_split=0.3  # setting split for traing and validation (70 and 30)
)

# training
trainGenerator = datagen.flow_from_directory(
    path,
    target_size=(224, 224),  # shape of images
    color_mode="rgb",  # coloured
    batch_size=32,  # number of samples processed in a single step
    class_mode="categorical",  # one-hot encoded labes for multi-class classification
    shuffle=True,  # shuffle images at each epoch
    subset='training'
)

validationGenerator = datagen.flow_from_directory(
    path,
    target_size=(224, 224),  # shape of images
    color_mode="rgb",  # coloured
    batch_size=32,  # number of samples processed in a single step
    class_mode="categorical",  # one-hot encoded labes for multi-class classification
    shuffle=False,  # no shuffle
    subset='validation'
)

# retrieving names of classes(Cheetah, Lion and Tiger)
names = list(trainGenerator.class_indices.keys())
numberOfClasses = len(names)
print("Classes:", names)

# using pretrained VGG16 model that acts as a feture extractor
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# freezing layers
base_model.trainable = False

model = keras.Sequential([
    base_model,  # feature extractor
    keras.layers.Flatten(),  # connecting
    keras.layers.Dense(256, activation='relu'),  # layer with 256 neurons
    keras.layers.Dense(128, activation='relu'),  # layer with 128 neurons
    keras.layers.Dense(64, activation='relu'),  # layer with 64 neurons
    keras.layers.Dropout(0.5),  # random dropping neurons after each iteration
    keras.layers.Dense(numberOfClasses, activation='softmax')  # output
])
# compile
model.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
# stop when validation accuracy isn't improving in last 5 epochs
earlyStopping = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True  # restore to when the validation accuracy was the best
)
# training
history = model.fit(trainGenerator, epochs=20, validation_data=validationGenerator, callbacks=[earlyStopping])
# accuracy
valLoss, valAcc = model.evaluate(validationGenerator, verbose=2)
print('\nValidation accuracy:', valAcc)
# saving the model
model.save("vgg16_classifier.h5")

# extracting true class labels
realLabels = validationGenerator.classes
# generate predictions
predictions = model.predict(validationGenerator)
predictedLabels = np.argmax(predictions, axis=1)

# classification report
report = classification_report(realLabels, predictedLabels, target_names=names)
print(report)

# calculating AUC()
binaryRealLabels = label_binarize(realLabels, classes=[0, 1, 2])
aucScores = {}
for i, name in enumerate(names):
    aucScores[name] = roc_auc_score(binaryRealLabels[:, i], predictions[:, i])

for name, auc in aucScores.items():
    print(f"{name}: {auc:.2f}")

# confusion matrix
matrix = confusion_matrix(realLabels, predictedLabels)
display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=names)
display.plot()
plt.show()