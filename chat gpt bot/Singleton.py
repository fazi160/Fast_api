""" any class that wants to implement singleton design pattern can use this class as its metaclass """

from abc import ABCMeta


class Singleton(ABCMeta, type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]