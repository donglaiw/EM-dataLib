import numpy as np

def strToArr(in_str, delim=',', func=int):
    # input string to int array
    return np.array([func(x) for x in in_str.split(delim)])

