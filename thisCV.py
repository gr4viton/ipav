import numpy as np
import cv2

import findHomeography as fh

from subprocess import Popen, PIPE

# from cv2 import xfeatures2d
# import common
import time
import sys
import os
import time
import enum

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# global variables

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# function definitions


def imclearborder(im, radius, buffer, mask):
    # Given a black and white image, first find all of its contours

    #todo make faster copping as buffer is always the same size!
    buffer = im.copy()
    # _, contours, hierarchy = cv2.findContours(buffer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = cv2.findContours(buffer, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # _, contours, hierarchy = cv2.findContours(buffer, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    # hierarchy = Next, Previous, First_Child, Parent
    # Get dimensions of image
    n_rows = im.shape[0]
    n_cols = im.shape[1]

    cTouching = []  # indexes of contours that touch the border
    cInsideTouching = [] # indexes that are inside of contours that touch the border

    # print('len contour',len(contours))
    # For each contour...
    for idx in np.arange(len(contours)):
        # if contour is not external continue (otherwise it cannot touoch border
        if hierarchy[0][idx][3] != -1:
            continue

        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= n_rows - 1 - radius and rowCnt < n_rows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= n_cols - 1 - radius and colCnt < n_cols)

            if check1 or check2:

                cTouching.append(idx)

                # add first children inside cInsideTouching
                # = first children is the other edge of external touching contour
                q = hierarchy[0][idx][2] # first child index
                if q != -1:
                    cInsideTouching.append(q)

                # as this contour is already added
                break


    # as the contours are CHAIN_APPROX for speed, this makes also the "noise" dissapear
    approx_correction_thickness = 3

    # create mask to delete (not touching the child contours insides)

    # make the most external contour black
    for idx in cTouching:
        col = 0
        cv2.drawContours(mask, contours, idx, col, thickness=-1)
        cv2.drawContours(mask, contours, idx, col, thickness=approx_correction_thickness )

    # make the inner contours visible
    for idx in cInsideTouching:
        col = 255
        cv2.drawContours(mask, contours, idx, color=col, thickness=-1)
        col = 0
        cv2.drawContours(mask, contours, idx, color=col, thickness=approx_correction_thickness )

    # mask2 = mask.copy()
    # cv2.dilate(mask,mask2)
    cv2.bitwise_and(mask, im, buffer)
    # imgBWcopy = imgBWcopy * 255
    # imgBWcopy = imgBWcopy
    return buffer



def extractContourArea(im_scene, external_contour):
    mask = np.uint8( np.zeros(im_scene.shape) )
    col = 1
    cv2.drawContours(mask, [external_contour], 0, col, -1)

    scene_with_tag = np.uint8( np.zeros(im_scene.shape) )
    cv2.bitwise_and(mask, im_scene.copy(), scene_with_tag)
    scene_with_tag = scene_with_tag * 255

    return scene_with_tag


def findTagsInScene(im_scene, model_tag):
    allowed_size = 800
    max_size = max(im_scene.shape)
    if max_size > allowed_size:
        # find tags in smaller image
        a = allowed_size/max_size
        im_scene_smaller = cv2.resize(im_scene,(0,0),fx=a,fy=a)
        markuped_scene, seen_tags = findTags(im_scene_smaller, model_tag)
        if seen_tags is None or len(seen_tags) == 0:
            markuped_scene, seen_tags = findTags(im_scene, model_tag)
    else:
        markuped_scene, seen_tags = findTags(im_scene, model_tag)

    return markuped_scene, seen_tags

def findTags(im_scene, model_tag):

    # first create copy of scene (not to be contoured)
    scene_markuped = im_scene.copy()


    # _, contours, hierarchy = cv2.findContours(im_scene.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    _, external_contours, hierarchy = cv2.findContours(im_scene.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    # _, external_contours, hierarchy = cv2.findContours(im_scene.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    # _, contours, hierarchy = cv2.findContours(im_scene.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )

    seen_tags = []

    # for every external contour area
    for external_contour in external_contours:

        # zero out everything but the tag area
        scene_with_tag = extractContourArea(im_scene, external_contour)

        # initialize observed tag
        observed_tag = fh.C_observedTag(scene_with_tag, external_contour, scene_markuped)


        # find out if the tag is in the area
        observed_tag.calculate(model_tag)

        # append it to seen tags list
        seen_tags.append(observed_tag)

    return scene_markuped, seen_tags

    # # for every external contour area
    # for q in np.arange(len(external_contours)):
    #
    #     # leave only the biggest contour
    #     # ?? - gets rid of the "noise"
    #
    #
    #     # # remove all inner contours
    #     # if hierarchy[0][q][3] != -1:
    #     #     continue
    #
    #     # zero out everything but the tag
    #     mask = np.uint8( np.zeros(im_scene.shape) )
    #     col = 1
    #     cv2.drawContours(mask, external_contours, q, col, -1)
    #
    #     scene_with_tag = np.uint8( np.zeros(im_scene.shape) )
    #     cv2.bitwise_and(mask, im_scene.copy(), scene_with_tag)
    #     scene_with_tag = scene_with_tag * 255
    #     scene_with_tag = extractContourArea(im_scene, external_contour)
    #
    #     # # find out euler number
    #     # _, tagCnt, hie = cv2.findContours(scene_with_tag.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     # if hie is None: continue
    #     # if len(hie) == 0: continue
    #     # hie = hie[0]
    #     # innerSquaresCount = 2
    #     # cntNumMin = 2  # case of joined inner squares
    #     # cntNumMax = 1 + innerSquaresCount
    #
    #     # if len(hie) >= cntNumMin and len(hie) <= cntNumMax:
    #     #     imTags.append(scene_with_tag)
    #     #     # imTags = imIn[y:y+h,x:x+w]
    #
    #     # if it sustains tha check of some kind
    #
    #     observed_tag = fh.C_observedTag(scene_with_tag, scene_markuped)
    #     # observed_tag = fh.C_observedTag(scene_with_tag, None)
    #     observed_tag.calculate(chain)
    #
    #     seen_tags.append(observed_tag)
    #
    # return scene_markuped, seen_tags

def bwareaopen(imgBW, areaPixels,col = 0):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, col, -1)
    return imgBWcopy

def threshIT(im, type):
    ## THRESH
    if type == cv2.THRESH_BINARY_INV or type == 1:
        _, th1 = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
        return th1
    elif type == cv2.ADAPTIVE_THRESH_MEAN_C or type == 2:
        th2 = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        return th2
    elif type == cv2.ADAPTIVE_THRESH_MEAN_C or type == 3:
        th3 = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        return th3

    ## OTSU
    elif type == 'otsu' or type == 4:
        ret, otsu = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu
    elif type == 'otsu_inv':
        ret, otsu = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return otsu


def add_text(im, text, col = 255, hw = (1, 20)):
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(im, text, hw , font, 1, 0, 5)
    cv2.putText(im, text, hw, font, 1, col)




class Step():
    """ Step class - this is usefull comment, literally"""
    def __init__(self, name, function):
        self.name = name
        self.function = function
        self.execution_time_len = 15
        self.execution_time = 0
        self.execution_times = []
        self.mean_execution_time = 0

    def run(self, data_prev):
        self.data_prev = data_prev
        self.user_input = False # e.g. from snippet or gui
        if self.user_input == True:
            data_prev[dd.kernel] = (42,42) # from user
            data_prev[dd.take_all_def] = False
        else:
            data_prev[dd.take_all_def] = True

        start = time.time()
        self.data_post = self.function(data_prev)
        end = time.time()
        self.add_exec_times(end-start)

        return self.data_post

    def get_info_string(self):
        return self.str_mean_execution_time()

    def add_exec_times(self, tim):
        if len(self.execution_times) > self.execution_time_len:
            self.execution_times.pop(0)
            self.add_exec_times(tim)
        else:
            self.execution_times.append(tim)
        self.mean_execution_time = np.sum(self.execution_times) / len(self.execution_times)

    def str_mean_execution_time(self, sufix=' ms'):
        return '{0:.2f}'.format(round(self.mean_execution_time * 1000,2)) + sufix


class AutoNumber(enum.Enum):
    def __new__(cls):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

class dd(AutoNumber):
    """
    DataDictParameterNames
    """
    im = ()
    kernel = ()
    resolution = ()
    sigma = ()
    take_all_def = ()
    fxfy = ()


class StepControl():

    buffer = None
    seen_tags = None
    mask_ones = None

    def recreate_mask(self,im ):
        if self.mask_ones is None or im.shape != self.mask_ones.shape:
            self.mask_ones = np.uint8( np.ones(im.shape) + 254)
            # print('recreated mask')

    def get_mask(self, im):
        self.recreate_mask(im)
        return self.mask_ones.copy()


    def recreate_buffer(self, im):
        if self.buffer is None:
            self.buffer = im.copy()
        else:
            if im.shape != self.buffer.shape:
                self.buffer = im.copy()
            # else:
            #     return self.buffer


    def get_buffer(self, im):
        if self.buffer is None:
            self.recreate_buffer(im)
        return self.buffer

    def add_available_step(self, name, function):
        self.available_steps[name] = Step(name, function)
        self.available_step_fcn[name] = function

    def select_steps(self, current_chain):
        self.chain = current_chain

        if current_chain.tag_search == True:
            def make_find_tags(im):
                markuped_scene, seen_tags = findTagsInScene(im.copy(), self.chain)
                self.seen_tags = seen_tags
                return markuped_scene

            # add find tag with currentyl selected current_chain
            findtag_name = 'findTags'
            self.available_steps.pop(findtag_name, None)
            self.add_available_step(findtag_name, make_find_tags)

        # create steps list for this current_chain algorithm
        self.steps = []

        # [self.steps.append(self.available_steps[step_name].copy()) for step_name in self.chain.step_names]
        # [self.steps.append(Step(step_name, self.available_step_fcn[step_name])) for step_name in self.chain.step_names]
        [self.steps.append(Step(step_name, self.available_steps[step_name].function)) for step_name in self.chain.step_names]


    def __init__(self, resolution_multiplier, current_chain):

        self.resolution_multiplier = resolution_multiplier
        self.define_available_steps()
        self.select_steps(current_chain)


    # def add_operation(self):
    #     pass

    def run_all(self, im):
        data = {}
        data[dd.im] = im
        # print(data)
        for step in self.steps:
            data = step.run(data)
        self.ret = data

    def step_all(self, im, resolution_multiplier):
        self.resolution_multiplier = resolution_multiplier
        self.run_all(im)


    def define_available_steps(self):
        """
        dd = data_dict
        """
        self.available_steps = {}
        self.available_step_fcn = {}


        def make_nothing(data):
            # print(data)
            im = data[dd.im]
            data[dd.resolution] = im.shape
            # if data["take_all_def"] == True or data[dd.kernel] == None:
            #     kernel = [5,5] # default kernel value
            #     data[dd.kernel] = kernel

            # im_out = gray(im, kernel)

            # data[dd.im] = im_out.copy()

            return data

        def make_gauss(data):
            im = data[dd.im]
            data[dd.resolution] = data[dd.im].shape
            take_def = data[dd.take_all_def] == True
            if take_def or data[dd.kernel] == None:
                kernel = (9,9) # default kernel value
                data[dd.kernel] = kernel
            if take_def or data[dd.sigma] == None:
                sigma = 1
                data[dd.sigma] = sigma

            im_out = cv2.GaussianBlur(im.copy(), kernel, sigmaX=sigma)

            data[dd.im] = im_out.copy() # ?? do i need a copy?
            return data

        # def make_gauss(im, a=5, sigma=1):
        #     return cv2.GaussianBlur(im.copy(), (a, a), sigmaX=sigma)

        def make_resize(data):
            im = data[dd.im]
            data[dd.resolution] = data[dd.im].shape
            take_def = data[dd.take_all_def] == True
            if take_def or data[dd.fxfy] == None:
                fxfy = [self.resolution_multiplier, self.resolution_multiplier]
                data[dd.fxfy] = fxfy

            im_out = cv2.resize(im.copy(), (0, 0), fx=fxfy[0], fy=fxfy[1])
            data[dd.im] = im_out.copy() # ?? do i need a copy?
            return data

        # def make_resize(im):
        #     return cv2.resize(im.copy(), (0, 0), fx=self.resolution_multiplier, fy=self.resolution_multiplier)



        def make_gray(im):
            return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        def make_clahe(im):
            return clahe.apply(im)

        def make_blur(im, a=75):
            return cv2.bilateralFilter(im, 9, a, a)

        def make_median(im, a=5):
            return cv2.medianBlur(im, a)

        def make_otsu(im):
            return threshIT(im,'otsu').copy()

        def make_otsu_inv(im):
            return threshIT(im,'otsu_inv').copy()

        def make_clear_border(im, width = 5):
            return imclearborder(im, width, self.get_buffer(im), self.get_mask(im))

        def make_remove_frame(im, width = 5, color = 0):
            a = width
            return cv2.copyMakeBorder(im[a:-a, a:-a], a, a, a, a,
                                      cv2.BORDER_CONSTANT, value=color)

        def make_invert(im):
            return (255 - im)

        def make_flood(im, color = 0):
            # im = im.copy()
            im = im.copy()
            h, w = im.shape[:2]
            a = 2
            mask = np.zeros((h + a, w + a), np.uint8)
            mask[:] = 0
            #seed = None
            seed = (0,0)
            rect = 4
            # rect = 8
            cv2.floodFill(im, mask, seed, color, 0, 255, rect)
            return im


        # Initiate ORB detector
        scoreType = cv2.ORB_FAST_SCORE
        # scoreType = cv2.ORB_HARRIS_SCORE
        orb = cv2.ORB_create(scoreType=scoreType)
        orb.setMaxFeatures(1000)
        # orb.setScaleFactor(1.2)
# 'setEdgeThreshold','setFastThreshold', \
# 'setFirstLevel', 'setMaxFeatures', \
# 'setNLevels', 'setPatchSize', \
# 'setScaleFactor', 'setScoreType',
#         setWTA_K

        # orb = cv2.ORB_create()
        # chainName = '2L'
        # chainName = 'c2'
        # chainName = 'standard'
        # chain = fh.Chain(chainName)
        # imTag = chain.imTag
        #
        #
        # kp_tag = orb.detect(imTag, None)
        #
        # # compute the descriptors with ORB
        # kp_tag, des_tag = orb.compute(imTag, kp_tag)
        #
        # def make_orb_2(im):
        #     # find the keypoints with ORB
        #     kp = orb.detect(im, None)
        #     # compute the descriptors with ORB
        #     kp, des = orb.compute(im, kp)
        #
        #     col = 142
        #     im_out = np.zeros(im.shape)
        #     flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + (
        #             cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        #
        #     des1 = des
        #     des2 = des_tag
        #
        #     # FLANN parameters
        #     FLANN_INDEX_KDTREE = 0
        #     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        #     search_params = dict(checks=50)   # or pass empty dictionary
        #
        #     flann = cv2.FlannBasedMatcher(index_params,search_params)
        #
        #     matches = flann.knnMatch(des1,des2,k=2)
        #
        #     # Need to draw only good matches, so create a mask
        #     matchesMask = [[0,0] for i in range(len(matches))]
        #
        #     # ratio test as per Lowe's paper
        #     for i,(m,n) in enumerate(matches):
        #         if m.distance < 0.7*n.distance:
        #             matchesMask[i]=[1,0]
        #
        #     draw_params = dict(matchColor = (0,255,0),
        #                        singlePointColor = (255,0,0),
        #                        matchesMask = matchesMask,
        #                        flags = 0)
        #
        #     flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT + cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
        #
        #     # Draw first 10 matches.
        #     sh = [im.shape[0] + imTag.shape[0], im.shape[1] + imTag.shape[1]]
        #     im_out = np.zeros(sh)
        #     cv2.drawMatches(im, kp, imTag, kp_tag, matches[:10], flags=flags, outImg=im_out)
        #
        #     return im_out
        #
        # def make_orb(im):
        #     # find the keypoints with ORB
        #     kp = orb.detect(im, None)
        #
        #     # compute the descriptors with ORB
        #     kp, des = orb.compute(im, kp)
        #
        #     # draw only keypoints location,not size and orientation
        #     # col = (0,255,0)
        #     col = 142
        #     im_out = np.zeros(im.shape)
        #     flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + (
        #             cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        #
        #     # draw only one keypoints set
        #     # cv2.drawKeypoints(im, kp_tag, im_out, color=col, flags=flags)
        #     # cv2.drawKeypoints(im, kp, im_out, color=col, flags=flags)
        #
        #
        #     # create BFMatcher object
        #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #
        #     # Match descriptors.
        #     matches = bf.match(des,des_tag)
        #
        #     # Sort them in the order of their distance.
        #     matches = sorted(matches, key = lambda x:x.distance)
        #
        #     flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT + cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
        #     # Draw first 10 matches.
        #     sh = [im.shape[0] + imTag.shape[0], im.shape[1] + imTag.shape[1]]
        #     im_out = np.zeros(sh)
        #     cv2.drawMatches(im, kp, imTag, kp_tag, matches[:10], flags=flags, outImg=im_out)
        #
        #     return im_out


        # def make_dif_laplace(im):



        # # Init SURF detector
        #
        # surf = cv2.xfeatures2d.SURF_create()
        #
        # def make_surf(im):
        #     kp = surf.detect(im, None)
        #
        #     col = 142
        #     im_out = np.zeros(im.shape)
        #     flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + (
        #             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #     cv2.drawKeypoints(im, kp, im_out, color=col, flags=flags)
        #
        #     return im_out
        #
        # # Init SIFT detector
        # sift = cv2.xfeatures2d.SIFT_create()
        #
        # def make_sift(im):
        #     kp = sift.detect(im, None)
        #
        #     col = 142
        #     im_out = np.zeros(im.shape)
        #     flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + (
        #             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #     cv2.drawKeypoints(im, kp, im_out, color=col, flags=flags)
        #
        #     return im_out
        #
        # freakExtractor = cv2.xfeatures2d.FREAK_create()
        # def make_freak(im):
        #     keypoints, descriptors = freakExtractor.compute(im, keypoints)
        #
        # # Initiate FAST object with default values
        # fast = cv2.FastFeatureDetector_create(nonmaxSuppression=0) # 8ms
        # fast = cv2.FastFeatureDetector_create(nonmaxSuppression=1) # 3ms
        # # fast.setBool('nonmaxSuppression',1)
        #
        #
        # def make_fast(im):
        #
        #     # find and draw the keypoints
        #     kp = fast.detect(im, None)
        #
        #     col = 255
        #     im_out = np.zeros(im.shape)
        #     flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + (
        #             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #
        #
        #     keypoints, descriptors = freakExtractor.compute(im, kp)
        #
        #     cv2.drawKeypoints(im, kp, im_out, color=col, flags=flags)
        #
        #
        #     return im_out

        def make_stack(im, code=cv2.COLOR_RGB2HLS_FULL):
            im_gray = make_gray(im)
            if code != 0:
                im_code = cv2.cvtColor(im, cv2.COLOR_RGB2HLS_FULL)
            else:
                im_code = im

            a, b, c = cv2.split(im_code)
            im1 = fh.joinIm([[a],[b]])

            im2 = fh.joinIm([[c], [im_gray]])
            im_out = fh.joinIm([[im1], [im2]], vertically=1)

            return im_out

        def make_hls_stack(im):
            im_gray = make_gray(im)
            im_hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS_FULL)
            h,l,s = cv2.split(im_hls)
            im1 = fh.joinIm([[h],[l]])


            im2 = fh.joinIm([[s], [im_gray]])
            # print(im1.shape,'ass',im2.shape)
            im_out = fh.joinIm([[im1], [im2]], vertically=1)
            # print(im_out.shape)
            return im_out

        def make_hls_saturation(im):
            # im_hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
            im_hls = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
            # return np.uint8( im_hls[:, :, 2] )
            return np.uint8( im_hls[:, :, 1] )

        def make_erase_color(im):
            im_gray = make_gray(im)
            gray_field = np.uint8( np.ones(im_gray.shape) + 0)
            # np.uint8( np.ones(im.shape) + 254)

            im_out = im_gray.copy()

            mask =  make_gauss(make_hls_saturation(im), a =9, sigma=13)
            mask = make_invert(mask)


            # _, th1 = cv2.threshold(mask, 127, 255, cv2.THRESH_TOZERO)
            # mask = th1

            # mask = np.round(mask)
            # cv2.bitwise_and(mask=mask, src1=im_gray, src2=gray_field, dst=im_out)

            alpha = 0.9
            beta = 0.1
            gamma = 0.0
            cv2.addWeighted(im_gray, alpha, mask, beta, gamma, im_out)

            # return mask
            # return gray_field
            # return im_gray
            return im_out

        def make_sobel(im, vertical=0, ksize=5):
            # out = im.copy()
            if vertical == 0:
                sob = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=ksize)
            else:
                sob = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=ksize)

            abs_sob = np.absolute(sob)
            return np.uint8(abs_sob)

        def make_laplacian(im):
            im2 = im.copy()
            return cv2.Laplacian(im2,cv2.CV_64F)




        sys.path.append("D:/DEV/PYTHON/pyCV/kivyCV_start/blender")

        # import blender_step
        #
        # bm = blender_step.blender_module()
        # bm.start_server()
        # bm.init_room()


# predavat dictionary - multiple possible images, text overlay
        def make_blender_cube(im):
            projections = 'contours of moving objects'



            imdir = bm.photogrammetry_object(projections)


            # imdir = os.path.abspath('D:\\DEV\\PYTHON\\pyCV\\kivyCV_start\\blender\\pic\\')
            dir_files = os.listdir(imdir)
            if dir_files == []:
                return im
            file_paths = [os.path.join(imdir, file) for file in dir_files]

            sorted_files = sorted(file_paths, key=os.path.getctime)
            # print('X'*111)

            latest_imfile = sorted_files [-1]
            # print('latest_imfile', latest_imfile)
            im = cv2.imread(latest_imfile)

            im_out = im
            return im_out

        def make_bounding_box_center():
            """
            from multiple camera segmented images - only bounding boxes of contours
            center of bounding box -> line intersection = center of object
            """

        def make_detect_red(im):
            # Convert BGR to HSV
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

            # define range of blue color in HSV
            blue = [[110,50,50],[130,255,255]]
            orange = [[24,50,50],[42,255,255]]
            red = [[0,80,80],[24,255,255]]

            color = red
            lower_blue = np.array(color[0])
            upper_blue = np.array(color[1])

            # Threshold the HSV image to get only blue colors
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            im_out = mask
            return im_out

        def make_detect_red2(im):
            # a = np.array([31, 67, 72], dtype=np.uint8)
            # b = np.array([28, 52, 76], dtype=np.uint8)
            x = 255/100
            a = np.array([24, 34*x, 26*x], dtype=np.uint8)
            b = np.array([39, 100*x, 100*x], dtype=np.uint8)

            # red/(r+g+b)
            #216

            lowerb, upperb = [a,b]
            # lowerb, upperb = [b,a]

            # im_float = np.array(im, dtype=np.float)
            im_float = im
            im_hsv = cv2.cvtColor(im_float, cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(im_hsv)
            print('hue', np.min(h), np.max(h))
            print(im_hsv.dtype)

            im_color = h
            # im_uint8 = np.array(im_color, dtype=np.uint8)
            im_uint8 = im_color

            blue, green, red = cv2.split(im)
            # im_color = red / (red/255 * green/255 * blue/255)
            # im_color = red / (red/255 + green/255 + blue/255)
            # im_color = red / (red + green + blue + 1) * 255
            im_color = red / (green + blue + 1) * 255
            im_color = 255 / (red + green + blue + 1)

            im_color = red / (red - green/2 - blue/2 + 1) * 255
            im_color = red - green/2 - blue/2
            im_color = 1/ (red - green/2 - blue/2)
            im_color = red/ (red - green - blue) * 255
            # im_color = red/ (red - green/2 - blue/2)
            im_color = red/ (red - green/3- blue/3)
            # im_color = 255 - red/ (red - green/2 - blue/2)

            # red_float = red
            # RMAX = 127
            #
            # red.convertTo(red_float, cv2.CV_32F)
            # red_float = red_float*RMAX/255+128;
            # red_float.convertTo(red,cv2.CV_8U);
            #
            # im_color = red
            # im_color = (red/255 + green/255 + blue/255)*255


            # im = im_uint8

            lowerb, upperb = [100,180]
            mask = cv2.inRange(im_color, lowerb, upperb)
            print('mask', np.min(mask), np.max(mask))
            im = mask * 1
            return im


        self.add_available_step('original', make_nothing)
        self.add_available_step('gray', make_gray)
        self.add_available_step('clahed', make_clahe)
        self.add_available_step('blurred', make_blur)

        self.add_available_step('gaussed', make_gauss)
        self.add_available_step('gauss', make_gauss)

        self.add_available_step('median', make_median)

        self.add_available_step('resize', make_resize)
        self.add_available_step('invert', make_invert)

        self.add_available_step('thresholded', make_otsu)
        self.add_available_step('thresholded inverted', make_otsu_inv)
        self.add_available_step('border touch cleared', make_clear_border)
        self.add_available_step('removed frame', make_remove_frame)
        self.add_available_step('flooded w/white', lambda im: make_flood(im, 255))
        self.add_available_step('flooded w/black', lambda im: make_flood(im, 0))

        # self.add_available_step('hls stack', make_hls_stack)
        self.add_available_step('hls stack', lambda im: make_stack(im, cv2.COLOR_RGB2HLS))
        self.add_available_step('rgb stack', lambda im: make_stack(im, 0))
        self.add_available_step('bgr stack', lambda im: make_stack(im, cv2.COLOR_RGB2BGR))
        self.add_available_step('hls saturation', make_hls_saturation)
        self.add_available_step('erase color', make_erase_color)


        # self.add_available_step('orb', make_orb)
        # self.add_available_step('sift', make_sift)
        # self.add_available_step('surf', make_surf)
        #
        # self.add_available_step('freak', make_freak)
        # self.add_available_step('fast', make_fast)
        self.add_available_step('sobel horizontal', lambda im: make_sobel(im, vertical=0, ksize=5))
        self.add_available_step('sobel vertical', lambda im: make_sobel(im, vertical=1, ksize=5))
        self.add_available_step('laplacian', make_laplacian)


        self.add_available_step('blender cube', make_blender_cube)
        self.add_available_step('detect red', make_detect_red)


        # self.available_steps.append(Step('original', make_nothing))
        # self.available_steps.append(Step('gray', make_gray))
        # self.steps.append(Step('clahed', make_clahe))
        # self.steps.append(Step('blurred', make_blur))
        # self.steps.append(Step('gaussed', make_gauss))
        #
        # self.available_steps.append(Step('resize', make_resize))
        #
        # self.available_steps.append(Step('tresholded', make_otsu))
        # self.available_steps.append(Step('border touch cleared', make_clear_border))
        # self.available_steps.append(Step('removed frame', make_remove_frame))
        # self.available_steps.append(Step('flooded w/white', lambda im: make_flood(im, 255)))
        # self.available_steps.append(Step('flooded w/black', lambda im: make_flood(im, 0)))



# def add_operation(operation_name, im_steps, im):
#     return im_steps.insert(0, [operation_name, [im]] )

# def loopCV(cap):
#     print("loopCV started")
#     while (True):
#         im = stepCV(cap)
#         cv2.imshow('image', im)
#         # How to end the loop
#         k = cv2.waitKey(30) & 0xff
#         if k == ord('q'):
#             break
#         if k == 27:
#             break
#         cv2.destroyAllWindows() # When everything done, release the capture

def waitKeyExit():
    while True:
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        if k == 27:
            break

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    ims = []

    cv2.namedWindow('image')

    tbValue = 3
    maxValue = 6
    # cv2.createTrackbar("trackMe", "image", tbValue, maxValue, update)
    # loopCV(cap)
    cap.release()
        # cnt = external_contours[q]

    cv2.destroyAllWindows()

