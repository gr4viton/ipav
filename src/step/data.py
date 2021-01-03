import numpy as np
from step.enums import DataDictParameterNames as dd


class StepData(dict):
    def __init__(self, *arg, **kw):
        super(StepData, self).__init__(*arg, **kw)

    def copy(self):
        dict_copy = self.__dict__.copy()
        # print(dict_copy.keys())
        if dd.im in dict_copy.keys():
            # if dd.im
            dict_copy[dd.im] = dict_copy[dd.im].copy()
            # print("making copy!")
        return dict_copy

    def copy_from(self, data):
        # self.__dict__ =
        if data is None:
            return StepData()
        for key, value in data.items():
            # print('key,value = {},{}'.format(key,value))
            if key == dd.im:
                if not dd.im in self.__dict__:
                    # if self.__dict__[dd.im] == None:
                    sh = data[dd.im].shape
                    self.__dict__[dd.im] = np.matrix(sh)
                    self.__dict__[dd.im] = data[dd.im].copy()
                    # data[dd.im].copy()
                    print(
                        "old = {} new = {}".format(
                            hex(id(self.__dict__[dd.im])),
                            hex(id(data[dd.im])),
                        )
                    )
                else:
                    # is there a function to just copy the data to the same address ? - would it be quicker?
                    self.__dict__[dd.im] = data[dd.im].copy()
                # elif:

            else:
                self.__dict__[key] = value
        # if dd.im in dict_copy.keys():
        #     # if dd.im
        #     dict_copy[dd.im] = dict_copy[dd.im].copy()
        #     # print("making copy!")
        # return dict_copy

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]
        # val = dict.__getitem__(self, key)
        return val

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def has_key(self, k):
        return self.__dict__.has_key(k)

    def pop(self, k, d=None):
        return self.__dict__.pop(k, d)

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()
