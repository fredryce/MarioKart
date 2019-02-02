import prediction
import cv2
import numpy as np
from goprocam import GoProCamera
from goprocam import constants
detection = prediction.OpenPose('mobile.pb', convert_csv=False)
gpCam = GoProCamera.GoPro()
cap = cv2.VideoCapture("udp://127.0.0.1:10000")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    result = detection.detect(frame)
    frame = detection.draw_humans(frame, result, imgcopy=False)


    cv2.imshow("GoPro OpenCV", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

