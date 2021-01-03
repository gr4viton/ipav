import threading


class LockedValue(object):
    """
    Thread safe lock
    """

    def __init__(self, init_val=None):
        self.lock = threading.Lock()
        self.val = init_val

    def __get__(self, obj, objtype):
        self.lock.acquire()
        if self.val != None:
            ret_val = self.val
        else:
            ret_val = None
        self.lock.release()
        # print('getting', ret_val)
        return ret_val

    def __set__(self, obj, val):
        self.lock.acquire()
        # print('setting', val)
        self.val = val
        self.lock.release()


class LockedNumpyArray(object):
    """
    Thread safe numpy array
    """

    def __init__(self, init_val=None):
        self.lock = threading.Lock()
        self.val = init_val

    def __get__(self, obj, objtype):
        self.lock.acquire()
        if self.val != None:
            # ret_val = self.val.copy()
            ret_val = self.val
        else:
            ret_val = None
        self.lock.release()
        # print('getting', ret_val)
        return ret_val

    def __set__(self, obj, val):
        self.lock.acquire()
        # print('setting', val)
        # self.val = val.copy() # ????????????????????????????????????????? do i need a copy??
        self.val = val
        self.lock.release()
