import enum

class AutoNumber(enum.Enum):
    def __new__(cls):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

class DataDictParameterNames(AutoNumber):
    """
    DataDictParameterNames == dd
    """
    # = ()
    im = ()
    info = ()
    info_text = ()
    # gauss ..
    kernel = ()
    resolution = ()
    sigma = ()
    take_all_def = ()
    fxfy = ()
    neighborhood_diameter = ()
    sigmaColor = ()
    sigmaSpace = ()
    # make_threshold
    thresh = ()
    maxVal = ()
    type = ()
    return_threshold_value = ()
    # make_sobel, make_laplacian
    ddepth = ()
    dx = ()
    dy = ()
    ksize = ()
    absolute = ()
    vertical = ()
    horizontal = ()
    # make_clear_border, make_color_edge
    width = ()
    value = ()
    # make_find_contours
    cnts = ()
    mode = ()
    method = ()
    thickness = ()
    color = ()
    colors = ()
    # make_convex_hull
    hull = ()
    hulls = ()
    # wait
    seconds = ()
    # captured source image
    captured = ()
    capture_control = ()
    stream = ()
    # res multiplier gui
    resolution_multiplier = ()
    # change name
    new_name = ()