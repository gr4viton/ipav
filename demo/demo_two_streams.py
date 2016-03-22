# http://stackoverflow.com/questions/29664399/reading-from-two-cameras-in-opencv-at-once
import cv2


cap0 = cv2.VideoCapture(0)
# cap0.set(3,160)
# cap0.set(4,120)

cap1 = cv2.VideoCapture(1)
# cap1.set(3,160)
# cap1.set(4,120)

ret0, frame0 = cap0.read()
assert ret0 # succeeds
ret1, frame1 = cap1.read()
assert ret1 # fails?!