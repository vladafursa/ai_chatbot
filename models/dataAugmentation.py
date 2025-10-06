from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import label_binarize

path = 'dataset'

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

model = keras.Sequential([
    keras.Input(shape=(224, 224, 3)),  # shape
    keras.layers.RandomFlip(mode="horizontal"),  # data augmentation
    keras.layers.RandomRotation(0.2),  # data augmentation
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(numberOfClasses, activation='softmax'),  # output
])

model.compile(optimizer='adam',
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# stop when validation accuracy isn't improving in last 5 epochs
earlyStopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True  # restore to when the validation accuracy was the best
)
# training
history = model.fit(trainGenerator, epochs=30, validation_data=validationGenerator, callbacks=[earlyStopping])
# accuracy
valLoss, valAcc = model.evaluate(validationGenerator, verbose=2)
print('\nValidation accuracy:', valAcc)

model.save("data_augmentation_classifier.h5")
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