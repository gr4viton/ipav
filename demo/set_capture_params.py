import cv2


def set_all_settings(capture):
    # width = 640
    # height = 480
    width, height = [640, 480]
    # width, height = [800, 600]
    width, height = [1024, 768]

    capture.set(3, width)
    capture.set(4, height)


#capture from camera at location 0
cap = cv2.VideoCapture(2)
#set the width and height, and UNSUCCESSFULLY set the exposure time
# cap.set(3,1280)
# cap.set(4,1024)

cap.set(3,640)
cap.set(4,480)
# set_all_settings(cap)
# cap.set(15, 0.1)


while True:
    ret, img = cap.read()
    cv2.imshow("input", img)
    #cv2.imshow("thresholded", imgray*thresh2)

    key = cv2.waitKey(10)
    if key == 27:
        break


cv2.destroyAllWindows()
cv2.VideoCapture(0).release()