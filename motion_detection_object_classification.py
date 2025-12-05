import cv2
import numpy as np
import imutils
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

# Initialize MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights="imagenet")

# Initialize video capture using your PC camera
cap = cv2.VideoCapture(0)

# Initialize the background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Resize the frame for faster processing and convert it to grayscale
    resized_frame = imutils.resize(frame, width=600)
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply the background subtractor to get the foreground mask
    fgmask = fgbg.apply(gray_frame)
    
    # Find contours in the foreground mask to detect objects
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        
        # Get bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(resized_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract the ROI (Region of Interest)
        roi = frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (224, 224))
        roi_array = np.expand_dims(roi_resized, axis=0)
        processed_roi = preprocess_input(roi_array)
        
        # Perform object classification using MobileNetV2
        predictions = model.predict(processed_roi)
        decoded_predictions = decode_predictions(predictions, top=1)[0][0]

        label = decoded_predictions[1]
        confidence = decoded_predictions[2]
        
        # Filter out less relevant objects
        if confidence > 0.5:
            if label in ["person", "tree", "screwdriver"]:
                text = f"{label} ({confidence*100:.2f}%)"
                cv2.putText(resized_frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow("Motion Detection and Object Classification", resized_frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
