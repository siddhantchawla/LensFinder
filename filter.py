from tensorflow import keras
from keras.models import load_model
import cv2
import numpy as np

# Load the model
my_model = load_model('my_model.h5')

# Cascading faces through xml file
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Defining the colour range for "Blue"
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

filters = ['images/sunglasses.png', 'images/sunglasses_2.png', 'images/sunglasses_3.jpg', 'images/sunglasses_4.png', 'images/sunglasses_5.jpg', 'images/sunglasses_6.png']
filterIndex = 4

camera = cv2.VideoCapture(0)

# Keep looping the video frames
while True:
    
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    frame2 = np.copy(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Add the 'Next Filter' button 
    frame = cv2.rectangle(frame, (500,10), (620,65), (235,50,50), -1)
    cv2.putText(frame, "NEXT FILTER", (512, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    faces = face_cascade.detectMultiScale(gray, 1.25, 6)

    # Determine which pixels fall within the blue boundaries and remove noise
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)

    # Find contours(blue bottle cap, in this case) in the image
    cnts, hierarchy = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Check to see if any contours were found
    if len(cnts) > 0:
     	# Sort the contours and find the largest one and assuming the largest to be the Blue cap
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Get the moments to calculate the center of the contour (in this case Circle)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        # Check if the Blue object is within the required region to activate the "Next Filter" trigger
        if center[1] <= 65:
            if 500 <= center[0] <= 620: # Next Filter
                filterIndex += 1
                filterIndex %= 6
                continue

    for (x, y, w, h) in faces:

        # Grab the face from the frame
        gray_face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]

        # Normalize the image to match the input format of the CNN - Range of pixels : [0, 1]
        gray_normalized = gray_face / 255

        # Resize it to 96x96
        original_shape = gray_face.shape # To keep the original shape safe
        face_resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized_copy = face_resized.copy()
        face_resized = face_resized.reshape(1, 96, 96, 1)

        keypoints = my_model.predict(face_resized) #Output of the CNN
        keypoints = keypoints * 48 + 48 #De-normalizing the output

        face_resized_color = cv2.resize(color_face, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized_color2 = np.copy(face_resized_color)

        # Pair the keypoints together
        points = []
        for i,co in enumerate(keypoints[0],0):
        	if i%2==0:
        		points.append((keypoints[0][i],keypoints[0][i+1]))

        # Add Filter to the frame
        sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)
        sunglass_width = int((points[7][0]-points[9][0])*1.1)
        sunglass_height = int((points[10][1]-points[8][1])/1.1)
        sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC)
        transparent_region = sunglass_resized[:,:,:3] != 0
        #Add only the glass to the image and not the background.
        face_resized_color[int(points[9][1]):int(points[9][1])+sunglass_height, int(points[9][0]):int(points[9][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]
        
        # Resize the face_resized_color image back to its original shape
        frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)

        cv2.imshow("Selfie Filters", frame)

        # To check the working of the model

        # Add KEYPOINTS to the frame2
        # for keypoint in points:
        #     cv2.circle(face_resized_color2, keypoint, 1, (0,255,0), 1)

        # frame2[y:y+h, x:x+w] = cv2.resize(face_resized_color2, original_shape, interpolation = cv2.INTER_CUBIC)
        # cv2.imshow("Facial Keypoints", frame2) 

    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
