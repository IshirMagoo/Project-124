# To Capture Frame
import cv2

# To process image array
import numpy as np

# import the tensorflow modules and load the model
import tensorflow as ts
model = ts.keras.models.load_model("keras_model.h5")

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		img = cv2.resize(frame,(224,224))
		img1 = np.array(img,dtype=np.float32)
		img1 = np.expand_dims(img1,axis=0)
		normalize = img1/255.0
		prediction = model.predict(normalize)
		print(prediction)
		rock = int(prediction[0][0]*100)
		paper = int(prediction[0][1]*100)
		scissor = int(prediction[0][2]*100)
		print(f"Rock: {rock} %, Paper: {paper} %, Scissor: {scissor} %")
		
		
		#resize the frame
		
		# expand the dimensions
		
		# normalize it before feeding to the model
		
		# get predictions from the model
		
		
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
