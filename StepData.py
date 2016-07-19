
from StepEnum import DataDictParameterNames as dd

class StepData(dict):
    def __init__(self,*arg,**kw):
        super(StepData, self).__init__(*arg, **kw)

    def copy(self):
        dict_copy = self.__dict__.copy()
        # print(dict_copy.keys())
        if dd.im in dict_copy.keys():
            # if dd.im
            dict_copy[dd.im] = dict_copy[dd.im].copy()
            # print("making copy!")
        return dict_copy

    def copy_insides(self):
        pass

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

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

    def values(self):
        return self.__dict__.values()
