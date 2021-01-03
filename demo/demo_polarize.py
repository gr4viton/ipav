import cv2
import numpy as np


class StepWidget:
    def convert_image(self, im_orig):
        im_type = im_orig.dtype
        if im_type != np.uint8:
            if im_type == np.float:
                im = self.polarize_image(im_orig)
            else:
                im = np.uint8(im_orig)
        else:
            im = im_orig

        return im

    def polarize_image(self, im):
        """create image with color polarization = negative values are with color1 and positive with color2"""

        if len(im.shape) == 2:
            pos = im.copy()
            pos[pos < 0] = 0
            pos = np.uint8(pos)

            neg = im.copy()
            neg[neg > 0] = 0
            neg = np.uint8(-neg)

            nul = np.uint8(0 * im)

            b = nul
            g = pos
            r = neg
            im_out = cv2.merge((b, g, r))
        else:
            im_out = im

        return im_out


if __name__ == "__main__":

    sw = StepWidget()

    cap = cv2.VideoCapture(0)

    while True:
        ret, im = cap.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        im = cv2.Sobel(im, cv2.CV_64F, dx=1, dy=0, ksize=5)

        im = sw.convert_image(im)

        cv2.imshow("input", im)
        # cv2.imshow("thresholded", imgray*thresh2)

        key = cv2.waitKey(10)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()
