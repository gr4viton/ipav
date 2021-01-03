import numpy as np
import cv2
import find_homeography as fh


def rgb_to_str(rgb):
    """Returns: string representation of RGB without alpha

    Parameter rgb: the color object to display
    Precondition: rgb is an RGB object"""
    return "[ " + str(rgb[0]) + ", " + str(rgb[1]) + ", " + str(rgb[2]) + " ]"


def imclearborder(im, radius, buffer, mask):
    # Given a black and white image, first find all of its contours

    # todo make faster copping as buffer is always the same size!
    buffer = im.copy()
    # _, contours, hierarchy = cv2.findContours(buffer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = cv2.findContours(
        buffer, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS
    )
    # _, contours, hierarchy = cv2.findContours(buffer, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    # hierarchy = Next, Previous, First_Child, Parent
    # Get dimensions of image
    n_rows = im.shape[0]
    n_cols = im.shape[1]

    cTouching = []  # indexes of contours that touch the border
    cInsideTouching = []  # indexes that are inside of contours that touch the border

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
            check1 = (rowCnt >= 0 and rowCnt < radius) or (
                rowCnt >= n_rows - 1 - radius and rowCnt < n_rows
            )
            check2 = (colCnt >= 0 and colCnt < radius) or (
                colCnt >= n_cols - 1 - radius and colCnt < n_cols
            )

            if check1 or check2:

                cTouching.append(idx)

                # add first children inside cInsideTouching
                # = first children is the other edge of external touching contour
                q = hierarchy[0][idx][2]  # first child index
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
        cv2.drawContours(
            mask, contours, idx, col, thickness=approx_correction_thickness
        )

    # make the inner contours visible
    for idx in cInsideTouching:
        col = 255
        cv2.drawContours(mask, contours, idx, color=col, thickness=-1)
        col = 0
        cv2.drawContours(
            mask, contours, idx, color=col, thickness=approx_correction_thickness
        )

    # mask2 = mask.copy()
    # cv2.dilate(mask,mask2)
    cv2.bitwise_and(mask, im, buffer)
    # imgBWcopy = imgBWcopy * 255
    # imgBWcopy = imgBWcopy
    return buffer


def extractContourArea(im_scene, external_contour):
    mask = np.uint8(np.zeros(im_scene.shape))
    col = 1
    cv2.drawContours(mask, [external_contour], 0, col, -1)

    scene_with_tag = np.uint8(np.zeros(im_scene.shape))
    cv2.bitwise_and(mask, im_scene.copy(), scene_with_tag)
    scene_with_tag = scene_with_tag * 255

    return scene_with_tag


def findTagsInScene(im_scene, model_tag):
    allowed_size = 800
    max_size = max(im_scene.shape)
    if max_size > allowed_size:
        # find tags in smaller image
        a = allowed_size / max_size
        im_scene_smaller = cv2.resize(im_scene, (0, 0), fx=a, fy=a)
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
    _, external_contours, hierarchy = cv2.findContours(
        im_scene.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
    )
    # _, external_contours, hierarchy = cv2.findContours(im_scene.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    # _, contours, hierarchy = cv2.findContours(im_scene.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )

    seen_tags = []

    # for every external contour area
    for external_contour in external_contours:

        # zero out everything but the tag area
        scene_with_tag = extractContourArea(im_scene, external_contour)

        # initialize observed tag
        observed_tag = fh.C_observedTag(
            scene_with_tag, external_contour, scene_markuped
        )

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


def bwareaopen(imgBW, areaPixels, col=0):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    _, contours, hierarchy = cv2.findContours(
        imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if area >= 0 and area <= areaPixels:
            cv2.drawContours(imgBWcopy, contours, idx, col, -1)
    return imgBWcopy


def threshIT(im, type):
    ## THRESH
    if type == cv2.THRESH_BINARY_INV or type == 1:
        _, th1 = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
        return th1
    elif type == cv2.ADAPTIVE_THRESH_MEAN_C or type == 2:
        th2 = cv2.adaptiveThreshold(
            im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return th2
    elif type == cv2.ADAPTIVE_THRESH_MEAN_C or type == 3:
        th3 = cv2.adaptiveThreshold(
            im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return th3

    ## OTSU
    elif type == "otsu" or type == 4:
        ret, otsu = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu
    elif type == "otsu_inv":
        ret, otsu = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return otsu


def add_text(im, text, col=255, hw=(1, 20)):
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(im, text, hw, font, 1, 0, 5)
    cv2.putText(im, text, hw, font, 1, col)
