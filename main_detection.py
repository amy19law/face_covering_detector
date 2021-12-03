# import packages
import cv2
import numpy as np
import imutils
import os
from os.path import dirname, join
import time
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load Required Models (Deep Neural Network Module & Caffe Models
prototext = r"deploy.prototxt.txt" # Define the Model Architecture (Layers)
caffemodel = r"res10_300x300_ssd_iter_140000.caffemodel" # Contains the Weights for the Actual Layers
detectFace = cv2.dnn.readNet(prototext, caffemodel)
# Load Face Covering Detector Model
detectFaceCovering = load_model("detection_model.model")
print("Loading Models")

def detection(detectFace, videoFrame, detectFaceCovering):

        # Initialise Variables
	faces = []
	locations = []
	predictions = []
        
	# Get the dimensions of frame & create a blob
	(height, width) = videoFrame.shape[:2]
	blob = cv2.dnn.blobFromImage(videoFrame, 1.0, (224, 224),(105, 175, 125))

	# Passing blob through the detectFace Network for Detection and Predictions
	detectFace.setInput(blob)
	faceCoveringDetections = detectFace.forward()

	# For Loop for Face Location Rectangle & Probability
	for i in range(0, faceCoveringDetections.shape[3]):
                
		# Extract Probability Connected to the Face Covering Detected
		probability = faceCoveringDetections[0, 0, i, 2]

		# Ensure that Weak Detections are Filtered Out
		if probability > 0.6:
                        
			# Create X,Y Coordinates for Rectangle around Detected Faces
			rect = faceCoveringDetections[0, 0, i, 3:7] * np.array([width, height, width, height])
			(x1, y1, x2, y2) = rect.astype("int")

			# Extract Face ROI, Preprocess, Convert to RGB & Resize
			person = videoFrame[y1:y2, x1:x2]
			person = img_to_array(person)
			person = preprocess_input(person)
			person = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
			person = cv2.resize(person, (224, 224))

			# Add the Face and Rectangle To Lists
			faces.append(person)
			locations.append((x1, y1, x2, y2))

	# If a Face is Detected, then Predictions can be made
	if len(faces) >= 1:
		faces = np.array(faces, dtype="float32")
		predictions = detectFaceCovering.predict(faces, batch_size=32)

	# Return the Face Locations & Corresponding Locations
	return (locations, predictions)

# Initialise Video Stream
print("Accessing & Starting Camera")
stream = VideoStream(src=0).start()

# While Video Stream is Active
while True:
        
	# Get the Frame from the Video Stream and Resize to 800 Width
	videoFrame = stream.read()
	videoFrame = imutils.resize(videoFrame, width=800)

	# Face Recognition & Detecting Face Covering
	(locations, predictions) = detection(detectFace, videoFrame, detectFaceCovering)

	# For Loop for the Detected Face Locations, Corresponding Locations & to Display Output
	for (rect, faceCoveringPrediction) in zip(locations, predictions):
		(faceCovering, withoutFaceCovering) = faceCoveringPrediction
		(x1, y1, x2, y2) = rect

		# Create Probability Label to be Displayed
		probabilityLabel = "Probability"
		probabilityLabel = "{}: {:.2f}%".format(probabilityLabel, max(faceCovering, withoutFaceCovering) * 100)

		# Determine which Detection Label to be Displayed
		detectionLabel = "Face Covering Detected" if faceCovering > withoutFaceCovering else "No Face Covering Detected"

		# Determine Colour of Rectangle & Text (Red = No Face Covering, Green = Face Covering)
		color = (0, 0, 255) if detectionLabel == "No Face Covering Detected" else (0, 255, 0)

		# Display the Labels and Rectangle around Detected Faces
		cv2.rectangle(videoFrame, (x1, y1), (x2, y2), color, 4)
		cv2.putText(videoFrame, detectionLabel, (310, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
		cv2.putText(videoFrame, probabilityLabel, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.60, color, 2)

		# Print into Shell Current Status of Detection & Probability
		print(detectionLabel)
		print(probabilityLabel)

	# Show Output
	cv2.imshow("Live Video Camera Stream", videoFrame)
	key = cv2.waitKey(10)
