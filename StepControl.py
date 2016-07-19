import numpy as np
import cv2

import findHomeography as fh

from subprocess import Popen, PIPE

# from cv2 import xfeatures2d
# import common
import time
import sys
import os

from StepEnum import DataDictParameterNames as dd
from StepData import StepData
from FcnAditional import *
from Step import Step
from StepEnum import DataDictParameterNames as dd

#from kivy.uix.slider import Slider
from kivy.uix.gridlayout import GridLayout

from kivy.properties import ObjectProperty, StringProperty, NumericProperty, BoundedNumericProperty

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# global variables

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# function definitions


class DetectColorControls(GridLayout):
    # selected_color = StringProperty('green')
    selected_color = 'green'

    def select_color(self, selected_color):
        self.selected_color = selected_color

    def get_control_values(self, data):
        data[dd.color_name] = self.selected_color
        print('selected_color= ', self.selected_color)
        return data

class ResolutionControls(GridLayout):
    slider_value = BoundedNumericProperty(0.5, min=0.05, max=2)
    slider = ObjectProperty()

    def get_control_values(self, data):
        # val = self.slider_value
        val = self.slider.value
        data[dd.fxfy] = [val, val]
        # print('getting data = ', val)
        return data

class StepControl():

    delimiter = ','
    strip_chars = ' \t'

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


    def get_controls(self, name):
        # print('Checking controls for step [{}]'.format(name))
        if name == 'resize':
            # print('returned controls !!!!!!!!!!!!!!!!!!!!!!')
            return ResolutionControls()
            # return None
        elif name == 'detect green':
            return DetectColorControls()
        else:
            return None


    def add_available_step(self, name, function, origin=None, controls=None):
        if not controls:
            controls = self.get_controls(name)


        if controls:
            print('Got controls for step [{}]'.format(name))
            print(controls)

        self.available_steps[name] = Step(name, function, controls=controls)
        # self.available_step_fcn[name] = function
        step = self.available_steps[name]
        if origin is not None:
            step.origin = origin
            self.available_steps[origin].synonyms.append(name)
        return name

    def add_synonyms(self, word_and_synonyms):
        delimiter = self.delimiter
        # strip_chars = self.strip_chars

        words = word_and_synonyms.split(delimiter)
        if len(words) > 1:
            word = words[0]
            synonym_word = words[1:]
            self.add_synonyms_separate(word, synonym_word)


    def add_synonyms_separate(self, word, synonyms_word):
        # self.available_steps = ['abs']
        # word = 'abs'
        # synonyms_word = 'abso, abss'
        if word in self.available_steps:
            function = self.available_steps[word].function
            delimiter = self.delimiter
            strip_chars = self.strip_chars

            synonyms = []
            # print(synonyms_word)
            if type(synonyms_word) == type([]):
                for synonym in synonyms_word:
                    if delimiter in synonym:
                        synonyms.extend(synonym.split(delimiter))
                    else:
                        synonyms.append(synonym)
            else:
                if delimiter in synonyms_word:
                    synonyms.extend(synonyms_word.split(delimiter))
                else:
                    synonyms.append(synonyms_word)



            synonyms = [self.add_available_step(synonym.strip(strip_chars), function, word) for synonym in synonyms]
            print('[{}] is now know also as [{}]'.format(word,synonyms))
        else:
            print('Cannot add synonyms to step_name [', word,
                  '] as it is not defined in available_steps')
        # if type(synonyms) == type(list):

    def select_steps(self, current_chain):
        self.chain = current_chain

        # legacy from findtag search in image
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

        hide_char = '.'
        info_char = 'i'
        for step_name in self.chain.step_names:
            narrowed = False
            # print(step_name)
            if step_name[0] == hide_char:
                step_name = step_name[1:]
                narrowed = True

            available_step = self.available_steps.get(step_name, None)
            if not available_step: # not known step name
                break

            new_step = Step(step_name, available_step.function, narrowed, available_step.controls)
            self.steps.append(new_step)
            # print(self.steps[-1].narrow)


    def __init__(self, resolution_multiplier, current_chain):

        self.resolution_multiplier = resolution_multiplier
        self.define_available_steps()
        self.select_steps(current_chain)


    # def add_operation(self):
    #     pass
    exc_delimiter = ':'
    def run_all(self, data):
        # data = StepData()
        # data[dd.im] = im
        data[dd.info] = False
        # print(data)
        for step in self.steps:
            try:
                data = step.run(data)
            except cv2.error  as ex:
                exc_string = ex.args[0]
                error_list = exc_string.split(self.exc_delimiter)
                error_string = error_list[-1]
                # error_number = error_list[-2]
                error_substring = error_string.split(' ')
                error_number_string = error_substring[1]
                error_number = int(error_number_string [1:-1])
                # ''.
                print(error_number)

                if error_number == -215:
                    print('Going to do automatic conversion to CV_8UC1 (functions abs and uint8)')
                    pos = 'pozice curent stepu'
                    self.add_new_steps(pos, 'uint8, abs')
                # errno, strerror = ex.args
                # print(errno, strerror)

                # exc_type, exc_obj, exc_tb = sys.exc_info()
                # print('Exception happened!\n', ex)
                # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # print(exc_type, fname, exc_tb.tb_lineno)

                # print('starttast')
                # [print(arg) for arg in ex.args]
                # if inst.args
            # data[dd.info] = False
        self.ret = data

    def step_all(self, data):
        # self.resolution_multiplier = data[dd.resolution_multiplier]
        self.run_all(data)


    def define_available_steps(self):
        """
        dd = data_dict
        """
        self.available_steps = {}
        # self.available_step_fcn = {}


        def add_default(data, param, value):
            if data[dd.take_all_def] == True or data[param] == None:
                data[param] = value
            return data[param]


        def make_nothing(data):
            # data[dd.resolution] = data[dd.im].shape
            stream = data[dd.stream]
            txt = 'S[{}]={}'.format(stream.source_id, stream.name)
            data[dd.new_name] = txt
            # info_text(data, txt)
            return data


        def make_set_source(data, source_id=0):
            # if source_id < len(data[dd.captured]):
                # data[dd.im] = data[dd.captured][source_id]
            stream = data[dd.capture_control].get_stream(source_id)
            if stream is None:
                print('ImageStreamControl with id=[{}] is nonexistent', source_id)
            else:
                data[dd.stream] = stream
                # print(stream.source_id)
                data[dd.im] = stream.frame
                txt = '|S[{}]={}|'.format(stream.source_id, stream.name)
                info_text(data,txt)

            return data

        def make_set_next_source(data):
            source_id = data[dd.capture_control].get_next_stream().source_id
            return self.make_set_source(data, source_id)
            # data[dd.stream] = stream
            # data[dd.im] = stream.frame


            # if stream is None:
            #     print('ImageStreamControl with id=[{}] is nonexistent', source_id)
            # else:
            #     data[dd.stream] = stream
            #     print(stream.source_id)
            #     data[dd.im] = stream.frame

            # return data


        def make_gauss(data, kernel=(5,5), sigma=(1,1)):
            kernel = add_default(data, dd.kernel, kernel)
            sigma = add_default(data, dd.sigma, sigma)
            data[dd.im] = cv2.GaussianBlur(data[dd.im], kernel, sigmaX=sigma[0], sigmaY=sigma[1])
            return data

        # def make_gauss(im, a=5, sigma=1):
        #     return cv2.GaussianBlur(im.copy(), (a, a), sigmaX=sigma)

        def make_resize(data, fx=0.5, fy=0.5):
            # resolution_multiplier = add_default(data,dd.resolution_multiplier, resolution_multiplier)
            fxfy = add_default(data, dd.fxfy, [fx, fy])
            data[dd.im] = cv2.resize(data[dd.im], (0, 0), fx=fxfy[0], fy=fxfy[1])
            return data

        # def make_resize(im):
        #     return cv2.resize(im.copy(), (0, 0), fx=self.resolution_multiplier, fy=self.resolution_multiplier)

        def make_gray(data):
            data[dd.im] = cv2.cvtColor(data[dd.im], cv2.COLOR_BGR2GRAY)
            return data

        # def make_color_transform(data)

        # def make_gray(im):
        #     return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


        def make_invert(data):
            data[dd.im] = (255 - data[dd.im])
            return data

        # def make_invert(im):
        #     return (255 - im)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        def make_clahe(data):
            data[dd.im] = clahe.apply(data[dd.im])
            return data

        # def make_clahe(im):
        #     return clahe.apply(im)

        def make_blur(data):
            neighborhood_diameter = add_default(data, dd.neighborhood_diameter, 5)
            sigmaColor = add_default(data, dd.sigmaColor, 100)
            sigmaSpace = add_default(data, dd.sigmaSpace, 100)
            data[dd.im] = cv2.bilateralFilter(data[dd.im], d=neighborhood_diameter,
                                              sigmaColor=sigmaColor,
                                              sigmaSpace=sigmaSpace)
            return data


        # def make_blur(im, a=75):
        #     return cv2.bilateralFilter(im, 9, a, a)

        def make_pause(data, seconds=1):
            seconds = add_default(data, dd.seconds, seconds)
            time.sleep(seconds)
            return data

        def make_sobel(data,
                       ksize=5, dx=0, dy=0, ddepth=cv2.CV_64F,
                       vertical=False, horizontal=False, absolute=False):

            ddepth = add_default(data, dd.ddepth, ddepth)
            dx = add_default(data, dd.dx, dx)
            dy = add_default(data, dd.dy, dy)
            ksize = add_default(data, dd.ksize, ksize)

            absolute = add_default(data, dd.absolute, absolute)
            vertical = add_default(data, dd.vertical, vertical)
            horizontal = add_default(data, dd.horizontal, horizontal)

            if vertical:
                dy = 1
            if horizontal:
                dx = 1

            im_out = cv2.Sobel(data[dd.im], ddepth=ddepth,
                                              dx=dx, dy=dy, ksize=ksize)

            if im_out is None:
                im_out = data[dd.im]

            if absolute:
                abs_sob = np.absolute(im_out)
                im_out = np.uint8(abs_sob)


            data[dd.im] = im_out
            return data


        # def make_sobel(im, vertical=0, ksize=5):
        #     # out = im.copy()
        #     if vertical == 0:
        #         sob = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=ksize)
        #     else:
        #         sob = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=ksize)
        #
        #     abs_sob = np.absolute(sob)
        #     return np.uint8(abs_sob)

        def make_laplacian(data):
            ddepth = add_default(data, dd.ddepth, cv2.CV_64F)
            ksize = add_default(data, dd.ksize, 5)
            data[dd.im] = cv2.Laplacian(data[dd.im], ddepth=ddepth, ksize=ksize)
            return data

        # def make_laplacian(im):
        #     im2 = im.copy()
        #     return cv2.Laplacian(im2,cv2.CV_64F)


        def make_median(data):
            ksize = add_default(data, dd.ksize, 5)
            data[dd.im] = cv2.medianBlur(data[dd.im], ksize=ksize)
            return data


        # def make_median(im, a=5):
        #     return cv2.medianBlur(im, a)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        def make_threshold(data):
            thresh = add_default(data, dd.thresh, 0)
            maxVal= add_default(data, dd.maxVal, 255)
            type = add_default(data, dd.type, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return_threshold_value, thresholded_im = cv2.threshold(data[dd.im], thresh, maxVal, type)
            data[dd.return_threshold_value] = return_threshold_value
            data[dd.im] = thresholded_im


            info_text(data, 'threshold = {}'. format(return_threshold_value))

            return data

        def make_otsu(data):
            data[dd.type] = cv2.THRESH_BINARY + cv2.THRESH_OTSU
            return make_threshold(data)

        def make_absolute(data):
            data[dd.im] = abs(data[dd.im])
            return data

        # def make_retype(data):
        #     np.uint8

        def make_uint8(data):
            data[dd.im] = np.uint8(data[dd.im])
            return data


        def make_clear_border(data, width=5):
            im = data[dd.im]
            width = add_default(data, dd.width, width)
            im_out = imclearborder(im, width, self.get_buffer(im), self.get_mask(im))
            data[dd.im] = im_out
            return data

        # def make_clear_border(im, width = 5):
        #     return imclearborder(im, width, self.get_buffer(im), self.get_mask(im))



        def make_otsu_inv(im):
            return threshIT(im,'otsu_inv').copy()


        def make_color_edge(data, width=5, value=0):
            print('geronima')
            im = data[dd.im]
            width = add_default(data, dd.width, width)
            value = add_default(data, dd.color, value)
            a = width
            im = cv2.copyMakeBorder(im[a:-a, a:-a], a, a, a, a,
                                      cv2.BORDER_CONSTANT, value=value)
            data[dd.im] = im
            return data


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




        # sys.path.append("D:/DEV/PYTHON/pyCV/kivyCV_start/blender")

        import blender_step

        self.bm = blender_step.blender_module()


# predavat dictionary - multiple possible images, text overlay
        def make_blender_cube(data):
            # prjs = 'contours of moving objects'
            if not self.bm.running:
                self.bm.start_server()
                time.sleep(1.5)

            if not self.bm.rooming:
                self.bm.init_room()
                time.sleep(1.5)


            # print('%'*42,'data[dd.hulls] = ', data[dd.hulls])

            imdir = self.bm.photogrammetry_object(data)

            # imdir = os.path.abspath('D:\\DEV\\PYTHON\\pyCV\\kivyCV_start\\blender\\pic\\')
            dir_files = os.listdir(imdir)
            if dir_files == []:
                return data
            file_paths = [os.path.join(imdir, file) for file in dir_files]

            sorted_files = sorted(file_paths, key=os.path.getctime)
            # print('X'*111)

            latest_imfile = sorted_files [-1]
            # print('latest_imfile', latest_imfile)
            im = cv2.imread(latest_imfile)
            if im is not None:
                print('loaded_image from folder blender/pic')
                data[dd.im] = im
            else:
                data[dd.im] = data[dd.im].copy()

            return data

        def make_bounding_box_center():
            """
            from multiple camera segmented images - only bounding boxes of contours
            center of bounding box -> line intersection = center of object
            """

        color_dict_hsv = {}
        color_dict_hsv['blue'] = [[110,50,50],[130,255,255]]
        # color_dict_hsv['green'] = [[50,110,50],[255,130,255]]
        color_dict_hsv['green1'] = [[35,80,50],[108,240,250]]

        color_dict_hsv['green'] = [[35,60,30],[108,240,255]]
        color_dict_hsv['orange'] = [[24,50,50],[42,255,255]]
        color_dict_hsv['red'] = [[0,80,80],[24,255,255]]

        def make_detect_color(data, color_name='red'):
            im = data[dd.im]
            # Convert BGR to HSV
            if len(im.shape) == 2:
                print('Cannot detect color on grayscale image!')
                return data

            color_name = add_default(data, dd.color_name, color_name)

            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

            # define range of blue color in HSV

            color = color_dict_hsv.get(color_name, None)
            if not color:
                return data

            lower_blue = np.array(color[0])
            upper_blue = np.array(color[1])

            # Threshold the HSV image to get only blue colors
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            im_out = mask
            data[dd.im] = im_out

            data[dd.new_name] = 'detect ' + color_name
            txt = 'HSV <' + str(color)  + '>'
            info_text(data, txt)
            return data

        def get_spaced_colors(n):
            if n == 0:
                return None
            max_value = 16581375 #255**3
            min_value = 255
            interval = int(max_value / n)
            colors = [hex(I)[2:].zfill(6) for I in range(min_value, max_value, interval)]

            return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

        def make_find_contours(data):
            im = data[dd.im]
            mode = add_default(data, dd.mode, cv2.RETR_TREE)
            # method = add_default(data, dd.method , cv2.CHAIN_APPROX_SIMPLE)
            method = add_default(data, dd.method , cv2.CHAIN_APPROX_NONE)

            third, cnts, hierarchy = cv2.findContours(im.copy(), mode=mode, method=method)

            info_text(data, '{} contours'. format(len(cnts)))

            data[dd.cnts] = cnts

            data[dd.im] = draw_cnts(data)
            return data


        def info_text(data, text):
            if data[dd.info] == False:
                data[dd.info] = True
                data[dd.info_text] = ''
            info_text = data.get(dd.info_text, '')
            data[dd.info_text] = info_text + text


        def draw_cnts(data, parameter=dd.cnts, contour_index=-1):
            im = data[dd.im]
            cnts = data[parameter]
            thickness =  add_default(data, dd.thickness , 1)

            cnts_count = len(cnts)
            if cnts_count == 0:
                return im
            colors = get_spaced_colors(cnts_count)
            data[dd.colors] = colors
            # color = add_default(data, dd.color, (0,255,0))

            shape = list(im.shape)
            if len(shape) < 3:
                shape.append(3)
            im_cnts = np.zeros(shape, np.uint8)
            # print(im_cnts.shape)
            # print(cnts)

            if cnts_count > 1:
                for (cnt, color) in zip(cnts, colors):
                    cv2.drawContours(im_cnts, cnt, contour_index, color, thickness)
            else:
                color = (0,255,0)
                cv2.drawContours(im_cnts, cnts, 0, color, thickness)

            return im_cnts

        def make_convex_hull(data):
            if data[dd.cnts] == None:
                # calc cnts
                data = make_find_contours(data)

            data[dd.info_text] += 'Convex hull'
            cnts = data[dd.cnts]
            if len(cnts) < 1:
                return data
            # print(len(cnts))

            # get biggest cnt
            len_max = 0
            for cnt in cnts:
                if len(cnt) > len_max:
                    len_max=len(cnt)
                    cnt_huge = cnt

            # cnt = cnts[0]
            cnt = cnt_huge

            # print(len(cnt))
            hull = cv2.convexHull(cnt)
            this_hull = [hull]
            data[dd.hull] = this_hull
            name = data[dd.stream].name

            # print(name)

            # if data.get(dd.hulls,] == None:
            if data.get(dd.hulls, None) is None:
                data[dd.hulls] = {name: this_hull}
            else:
                data[dd.hulls][name] = this_hull

            # print(name,data[dd.hulls][name])
            # data[dd.hulls].insert(id, [hull])
            # data[dd.hulls][id] = [hull]

            # print(len(hull))
            data[dd.im] = draw_cnts(data, dd.hull, 0)

            # print("aa",hull)

            return data



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
        self.add_synonyms('original')

        self.add_available_step('gray', make_gray)
        self.add_available_step('clahe', make_clahe)
        self.add_synonyms('clahe, clahed')

        self.add_available_step('blur', make_blur)
        self.add_synonyms('blur, blurred')

        self.add_available_step('gauss', make_gauss)
        self.add_synonyms('gauss, gaussed')
        self.add_available_step('mega gauss', lambda d: make_gauss(d, kernel=(7,7), sigma=(3,3)))

        self.add_available_step('median', make_median)

        self.add_available_step('resize', make_resize)
        self.add_available_step('invert', make_invert)

        self.add_available_step('threshold', make_threshold)
        self.add_synonyms('threshold, thresh, thresholded')
        self.add_available_step('otsu', make_otsu)


        self.add_available_step('abs', make_absolute)
        self.add_synonyms('abs, absolute')

        self.add_available_step('uint8', make_uint8)
        self.add_synonyms('uint8, ubyte, dec, decimate, uint')

        self.add_available_step('sobel', lambda d: make_sobel(d, vertical=True, horizontal=True))
        self.add_available_step('sobh', lambda d: make_sobel(d, horizontal=True))
        self.add_available_step('sobv', lambda d: make_sobel(d, vertical=True))
        self.add_synonyms('sobel, sob, sobvh, sobhv')
        self.add_synonyms('sobh, sobelh, sobel horizontal')
        self.add_synonyms('sobv, sobelv, sobel vertical')


        self.add_available_step('pause', lambda d: make_pause(d, seconds=1))
        self.add_synonyms('pause, pause 5')

        self.add_available_step('laplacian', make_laplacian)
        self.add_synonyms('laplacian, laplace, lap, lapla')

        self.add_available_step('contours', make_find_contours)
        self.add_synonyms('contours, find contours, cnt, cnts, contour')


        self.add_available_step('convex hull', make_convex_hull)
        self.add_synonyms('convex hull, hull')

        self.add_available_step('blender', make_blender_cube)
        self.add_synonyms('blender, blend, blender cube')

        # self.add_available_step('detect red', make_detect_color)
        self.add_available_step('detect red', lambda d: make_detect_color(d, color_name='red'))
        self.add_available_step('detect orange', lambda d: make_detect_color(d, color_name='orange'))
        self.add_available_step('detect blue', lambda d: make_detect_color(d, color_name='blue'))
        self.add_available_step('detect green', lambda d: make_detect_color(d, color_name='green'))

        self.add_available_step('source', make_set_source)
        self.add_available_step('source1', lambda d: make_set_source(d, source_id=1))
        self.add_available_step('source2', lambda d: make_set_source(d, source_id=2))
        self.add_available_step('source3', lambda d: make_set_source(d, source_id=3))
        self.add_available_step('source4', lambda d: make_set_source(d, source_id=4))
        self.add_available_step('source5', lambda d: make_set_source(d, source_id=5))
        self.add_synonyms('source, set source, source0')

        self.add_available_step('next source', make_set_next_source)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        self.add_available_step('clear border', make_clear_border)
        self.add_synonyms('clear border, border touch cleared, remove border touching')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.add_available_step('color edge', make_color_edge)
        self.add_synonyms('color edge, remove frame, color frame, color')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NOT YET
#         self.add_available_step('flooded w/white', lambda im: make_flood(im, 255))
#         self.add_available_step('flooded w/black', lambda im: make_flood(im, 0))
#
#
#         self.add_available_step('thresholded inverted', make_otsu_inv)
#
#         # self.add_available_step('hls stack', make_hls_stack)
#         self.add_available_step('hls stack', lambda im: make_stack(im, cv2.COLOR_RGB2HLS))
#         self.add_available_step('rgb stack', lambda im: make_stack(im, 0))
#         self.add_available_step('bgr stack', lambda im: make_stack(im, cv2.COLOR_RGB2BGR))
#         self.add_available_step('hls saturation', make_hls_saturation)
#         self.add_available_step('erase color', make_erase_color)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # self.add_available_step('orb', make_orb)
        # self.add_available_step('sift', make_sift)
        # self.add_available_step('surf', make_surf)
        #
        # self.add_available_step('freak', make_freak)
        # self.add_available_step('fast', make_fast)



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

