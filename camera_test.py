import sys
import cv2

if len(sys.argv) >= 2:
    id = int(sys.argv[1])
else:
    id = 0
capture = cv2.VideoCapture(id)

while(True):
    ret, frame = capture.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
