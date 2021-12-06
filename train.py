# Created by Amy Law
# Import Packages & Modules
from imutils import paths
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Location of Files containing Images
location = r"C:\Users\amylaw\Documents\Intelligent Systems Project\dataset\"
files = ["with", "without"]

# Get Images List & then Initialise the List of Data
print("Loading Images")
data = []
labels = []

# Initialise Learning Rate, Total Amount of Epochs to Train & Batch Size
learningRate = 1e-4
epochsNumber = 20
batchSize = 32

for category in files:
    path = os.path.join(location, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size = (224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)
    	data.append(image)
    	labels.append(category)

# Encode Labels
labelBin = LabelBinarizer()
labels = labelBin.fit_transform(labels)
labels = to_categorical(labels)
data = np.array(data, dtype="float32")
labels = np.array(labels)
(x1, x2, y1, y2) = train_test_split(data, labels, test_size = 0.20, stratify = labels, random_state = 42)

# Construct Training Image Generator for Data Augmentation
augmentation = ImageDataGenerator(rotation_range = 20, zoom_range = 0.15, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.15, horizontal_flip = True, fill_mode = "nearest")

# Load the MobileNetV2 Network
baseModel = MobileNetV2(weights = "imagenet", include_top = False, input_tensor = Input(shape=(224, 224, 3)))

# Created by Amy Law

# Construct the main model that will be placed on top of the base model
mainModel = baseModel.output
mainModel = AveragePooling2D(pool_size=(7, 7))(mainModel)
mainModel = Flatten(name="flatten")(mainModel)
mainModel = Dense(128, activation="relu")(mainModel)
mainModel = Dropout(0.5)(mainModel)
mainModel = Dense(2, activation="softmax")(mainModel)

# Place the Main Model on Top of the Base Model in order to Produce the Final Model
model = Model(inputs = baseModel.input, outputs = mainModel)

# For Loop over all layers in the base model and freeze them
for layer in baseModel.layers:
	layer.trainable = False

# Compile Model
opt = Adam(lr = learningRate, decay = learningRate / epochsNumber)
model.compile(loss="binary_crossentropy", optimizer = opt, metrics=["accuracy"])

# Training
print("Training Started")
networkHead = model.fit(augmentation.flow(x1, y1, batchSize), steps_per_epoch = len(x1) // batchSize, validation_data = (x2, y2), validation_steps = len(x1) // batchSize, epochs = epochsNumber)

# Predictions on the Testing Set
predictionIndexs = model.predict(x2, batchSize)

# Index of Label with Corresponding Biggest Predicted Probability
predictionIndexs = np.argmax(predictionIndexs, axis = 1)

# Classification Report
print(classification_report(y2.argmax(axis = 1), predictionIndexs, target_names = labelBin.classes_))

# Save Model to be use in Detection Program
print("Saving Model")
model.save("detection_model.model", save_format="h5")

# Created by Amy Law
