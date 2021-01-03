import numpy as np
import cv2
from enum import Enum

# import matplotlib.pyplot as plt
# from itertools import cycle
# import howToFindSqure as sq


class Error(Enum):
    flawless = 0
    no_square_points = 1
    no_inverse_matrix = 2
    no_tag_rotations_found = 3
    rotation_uncertainity = 4
    external_contour_error = 5
    contour_too_small = 6
    square_points_float_nan = 7


import time


class C_observedTag:
    # static class variable - known Tag
    # tagModels = loadTagModels('2L')

    def __init__(self, imTagInScene, external_contour, scene_markuped):
        self.imScene = imTagInScene  # image of scene in which the tag is supposed to be
        self.imWarped = None  # ground floor image of tag transformed from imScene
        # self.tag_warped = None # warped tag image into chain space

        self.dst_pts = None  # perspectively deformed detectionArea square corner points
        self.mWarp2tag = (
            None  # transformation matrix from perspective scene to ground floor tag
        )
        self.mWarp2scene = (
            None  # transformation matrix from ground floor tag to perspective scene
        )
        self.cntExternal = external_contour  # detectionArea external contour
        self.mu = None  # all image moments
        self.mc = None  # central moment
        self.rotation = None  # square symbols in symbolArea check possible rotations similar to cTagModel - [0,90,180,270]deg
        self.error = Error.flawless  # error
        self.scene_markuped = scene_markuped  # whole image to print additive markups of this observed tag to

        self.color_corners = 160
        self.color_centroid = 180

        # self.external_contour_approx = cv2.CHAIN_APPROX_SIMPLE
        # self.external_contour_approx = cv2.CHAIN_APPROX_NONE
        # self.external_contour_approx = cv2.CHAIN_APPROX_TC89_L1
        self.external_contour_approx = cv2.CHAIN_APPROX_TC89_KCOS

        self.verbatim = False  # if set to True the set_error function would print findTag error messages in default output stream

        self.minimum_contour_length = 4

    def calcMoments(self):  # returns 0 on success
        self.mu = cv2.moments(self.cntExternal)
        if self.mu["m00"] == 0:
            return 1
        self.mc = np.float32(getCentralMoment(self.mu))
        return 0

    def calcExternalContour(self):  # returns 0 on success
        _, contours, hierarchy = cv2.findContours(
            self.imScene.copy(), cv2.RETR_EXTERNAL, self.external_contour_approx
        )
        if len(contours) != 0:
            self.cntExternal = contours[0]
        return self.calcMoments()

    def getExternalContour(self, imScene):
        _, contours, hierarchy = cv2.findContours(
            imScene.copy(), cv2.RETR_EXTERNAL, self.external_contour_approx
        )
        return contours[0]

    def addExternalContour(self, cntExternal):  # returns 0 on success
        self.cntExternal = cntExternal
        return self.calcMoments()

    def set_error(self, error):
        self.error = error
        if self.verbatim == True:
            if error == Error.no_square_points:
                print("Could not find square in image")
            elif error == Error.no_inverse_matrix:
                print(
                    "Cannot create inverse matrix. Singular warping matrix. Probably bad tag detected!"
                )

        return self.error

    def findWarpMatrix(self, model_tag):  # returns 0 on succesfull matching

        if self.findSquare(model_tag) != Error.flawless:
            return self.error

        drawCentroid(
            self.scene_markuped, self.cntExternal, self.color_centroid
        )  # DRAW centroid

        # self.mWarp2tag, mask= cv2.findHomography(src_pts, self.dst_pts, cv2.RANSAC, 5.0)

        # method = cv2.LMEDS
        src_pts = model_tag.ptsDetectArea
        self.mWarp2scene, _ = cv2.findHomography(
            src_pts,
            self.dst_pts,
        )

        # get inverse transformation matrix
        try:
            self.mWarp2tag = np.linalg.inv(self.mWarp2scene)
        except:
            # raise Exception('Cannot calculate inverse matrix.')
            # print("Cannot create inverse matrix. Singular warping matrix. Probably bad tag detected!")
            return self.set_error(Error.no_inverse_matrix)

        self.imWarped = self.drawSceneWarpedToTag(model_tag)
        self.addWarpRotation(model_tag)
        self.imWarped = self.drawSceneWarpedToTag(model_tag)

        return self.error

    def addWarpRotation(self, model_tag):

        # find out if it is really a tag
        if model_tag.checkType == "symbolSquareMeanValue":

            imSymbolArea = model_tag.symbolArea.getRoi(self.imWarped)

            imSymbolSubAreas = []
            for area in model_tag.symbolSubAreas:
                imSub = area.getRoi(imSymbolArea)
                imSymbolSubAreas.append(imSub)

            squareMeans = model_tag.getSquareMeans(imSymbolSubAreas)
            # print(squareMeans)

            self.rotation = []
            for modelCode in model_tag.rotatedModelCodes:
                if modelCode == squareMeans:  # * 1
                    self.rotation.append(1)
                else:
                    self.rotation.append(0)

            # print(self.rotation)
            if sum(self.rotation) == 0:
                return self.set_error(Error.no_tag_rotations_found)
            if sum(self.rotation) > 1:
                return self.set_error(Error.rotation_uncertainity)

            self.rotIdx = np.sum([i * self.rotation[i] for i in range(0, 4)])
            self.mWarp2tag = matDot(model_tag.mInvRotTra[self.rotIdx], self.mWarp2tag)
        # thresholded element-wise addition
        # procentual histogram - of seenTag vs of tagModel

        return self.error

    def findSquare(self, model_tag):  # returns 0 on succesfull findings

        cnt = self.cntExternal
        if len(cnt) < self.minimum_contour_length:
            # print('aa',cnt)
            return self.set_error(Error.contour_too_small)

        corner_pts = self.findApproxPolyDP(cnt)
        # corner_pts = cornerSubPix #- shiThomasi

        # corners from FAST and then findStableLineIntersection
        # contours inter and outer energies - only lines
        # houghlines

        # tims[-1].stop()
        # print('times','| '.join([tim.last() for tim in tims]))

        if corner_pts is None or len(corner_pts) != 4:
            return self.set_error(Error.no_square_points)

        # for corner_pt in corner_pts:
        #     for z in corner_pt:
        #         if z is float('nan'):
        #             self.set_error(Error.no_square_points)

        [
            self.set_error(Error.square_points_float_nan)
            for corner_pt in corner_pts
            for z in corner_pt
            if np.isnan(z)
        ]
        if self.error != Error.flawless:
            return self.error

        self.dst_pts = np.array(corner_pts)
        drawDots(
            self.scene_markuped, self.dst_pts, self.color_corners
        )  # draw corner points
        return self.error

    def findApproxPolyDP(self, cnt):
        def angle_cos(p0, p1, p2):
            d1, d2 = (p0 - p1).astype("float"), (p2 - p1).astype("float")
            return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > 10 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max(
                [
                    angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4])
                    for i in range(4)
                ]
            )
            # if max_cos < 0.1:
            return cnt

    def calculate(self, model_tag):
        self.findWarpMatrix(model_tag)
        # if self.findWarpMatrix(chain) == Error.flawless:
        # self.tag_warped = self.drawSceneWarpedToTag(chain)

    def drawTagWarpedToScene(self, imTag, imScene):
        h, w = imTag.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst = cv2.perspectiveTransform(pts, self.mWarp2tag)
        return cv2.polylines(imScene, [np.int32(dst)], True, 128, 3, cv2.LINE_8)

    def drawSceneWarpedToTag(self, model_tag):
        # print self.mInverse
        return cv2.warpPerspective(
            self.imScene,
            self.mWarp2tag,
            model_tag.imTagDetect.shape,
            flags=cv2.INTER_LINEAR,
        )
        # , , cv2.BORDER_CONSTANT)


class Timeas:
    def __init__(self, type="s2ms"):
        self.start()
        self.set_output_type(type)

    def set_output_type(self, type):
        self.type = type

    def start(self):
        self.time_start = time.time()

    def stop(self):
        self.time_end = time.time()
        self.time_last = self.time_end - self.time_start

    def now(self):
        self.stop()
        return self.last()

    def print_last(self):
        print(self.last())

    def print(self):
        self.stop()
        print(self.last())

    def last(self):
        if self.type == "s2ms":
            return "{00:.2f} ms".format(round(self.time_last * 1000, 2))


class C_area:
    def __init__(self, hw, tl):
        self.hw = hw
        self.tl = tl

    def getRoi(self, im):
        return im[
            self.tl[0] : self.tl[0] + self.hw[0], self.tl[1] : self.tl[1] + self.hw[1]
        ]

    def getSubAreas(self, rows, cols):
        # one cell dimensions
        hSub = int(self.hw[0] / rows)
        wSub = int(self.hw[1] / cols)

        # border pixels vertical
        hSubMulti = hSub * rows
        if hSubMulti < self.hw[0]:
            # must append - > append to the last one
            hDiff = self.hw[0] - hSubMulti
        else:
            hDiff = 0

        # border pixels horizontal
        wSubMulti = wSub * cols
        if wSubMulti < self.hw[1]:
            # must append - > append to the last one
            wDiff = self.hw[1] - wSubMulti
        else:
            wDiff = 0

        # create the subareas
        aSubs = []
        hw = (hSub, wSub)
        for iRow in range(0, rows):
            for iCol in range(0, cols):
                tl = (iRow * hSub, iCol * wSub)
                aSub = C_area(hw, tl)
                if iRow == rows - 1:
                    if iCol == cols - 1:
                        aSub = C_area((hSub + hDiff, wSub + wDiff), tl)
                    else:
                        aSub = C_area((hSub, wSub + wDiff), tl)
                if iCol == cols - 1:
                    aSub = C_area((hSub, wSub + wDiff), tl)
                # print aSub.hw
                # print aSub.tl
                aSubs.append(aSub)

        return aSubs

    # def __init__(self, hw, tl, tlFromUpperHW = False):
    #     self.hw = hw
    #     if tlFromUpperHW == False:
    #         self.tl = tl
    #     else:
    #         self.tl = self.getTopLeftCentered(tl)


#
#     def __init__(self, hw, hwWhole):
#         self.height = h
#         self.width = w
#         [self.top, self.left] = getTopLeftCentered()
#
#     def getTopLeftCentered(self,hwUpper):
#         return self.


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def readIm(pre, tag):
    dIm = "./pic/"
    fIm = pre + "_" + tag + ".png"
    im = cv2.imread(dIm + fIm, 0)
    if im is not None:
        print("Loaded image: [" + fIm + "] = " + str(im.shape))
    return im


def getBoxCorners(boxOffset, boxSide):
    aS = boxOffset
    aB = boxOffset + boxSide
    pts = [[aS, aS], [aS, aB], [aB, aB], [aB, aS]]
    return np.float32(pts)


#
# def read_model_tag(strTag):
#     print(strTag)
#     cTag = C_chain(strTag)
#     return cTag


def makeBorder(im, bgColor):
    bs = max(im.shape)
    im = cv2.copyMakeBorder(im, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=bgColor)
    return im, bs


def makeLaplacian(im):
    return np.uint8(np.absolute(cv2.Laplacian(im, cv2.CV_64F)))


def joinTwoIm(imBig, imSmall, vertically=0, color=0):
    diff = imBig.shape[vertically] - imSmall.shape[vertically]
    if vertically == 0:
        imEnlarged = cv2.copyMakeBorder(
            imSmall, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=color
        )
        return np.hstack([imBig, imEnlarged])
    else:
        imEnlarged = cv2.copyMakeBorder(
            imSmall, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=color
        )
        return np.vstack([imBig, imEnlarged])


def joinIm(ims, vertically=0, color=0):
    imLast = []
    for im in ims:
        im = im[0]
        if imLast != []:
            # print(vertically)
            if (im.shape[vertically] - imLast.shape[vertically]) > 0:
                # im is bigger
                imLast = joinTwoIm(im, imLast, vertically, color)
            else:
                imLast = joinTwoIm(imLast, im, vertically, color)
        else:
            imLast = im

    return np.array(imLast)


def colorifyGray(im):
    return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)


def drawContour(im, cnt, color=180, thickness=1):
    cv2.drawContours(im, cnt, 0, color, thickness)


def drawDots(im, dots, numbers=1):
    i = 0
    for dot in dots:
        pt = [int(dot[0]), int(dot[1])]
        # col = (255, 0, 0)
        col = 180
        sh_max = np.max(im.shape)
        # radius = np.int(sh_max  / 40)
        radius = 1
        thickness = np.int(sh_max / 140)
        cv2.circle(im, tuple(pt), radius, col, thickness)
        numbers = 1
        if numbers == 1:
            # font = cv2.FONT_HERSHEY_SIMPLEX
            font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            cv2.putText(
                im, str(i), tuple([d + 10 for d in pt]), font, 1, 0, thickness + 2
            )
            cv2.putText(
                im, str(i), tuple([d + 10 for d in pt]), font, 1, 255, thickness
            )
        i += 1
    return im


def drawBoundingBox(im, cnt, color=255, lineWidth=1):
    # non-rotated boundingbox
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(im, (x, y), (x + w, y + h), color, lineWidth)


def drawRotatedBoundingBox(im, cnt, color=255, lineWidth=1):
    # draw rotated minAreaRect boundingBox
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    int_box = np.int0(box)
    cv2.drawContours(im, [int_box], 0, color, 1)


def drawCentroid(im, cnt, color=255):
    mu = cv2.moments(cnt)
    mc = getCentralMoment(mu)
    cv2.circle(im, tuple(int(i) for i in mc), 4, color, -1, 8, 0)


def getCentralMoment(mu):
    if mu["m00"] == 0:
        raise Exception("Moment of image m00 is zero. Could not count central moment!")
    return [mu["m10"] / mu["m00"], mu["m01"] / mu["m00"]]


def matDot(A, B):
    C = np.eye(3, 3)
    np.dot(A, B, C)
    return C


if __name__ == "__main__":

    # cv2.destroyAllWindows()

    pass


def waitKeyExit():
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == 27:
            break
