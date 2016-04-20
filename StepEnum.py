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
    im = ()
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