import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('1117.mp4')  # Replace '1117.mp4' with the path to your video file

# Define the vertices of the ROI polygon
vertices = np.array([(125, 536),
                          (598, 278),
                          (839, 366),
                          (798, 506),
                          (586, 606),
                          (188, 604)])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Create a mask with the same size as the frame
    mask = np.zeros_like(frame)

    # Draw the ROI polygon on the mask with white color
    cv2.fillPoly(mask, [vertices], (255, 255, 255))

    # Use bitwise AND operation to combine the mask with the frame
    result = cv2.bitwise_and(frame, mask)

    # Display the result
    cv2.imshow('Result', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()