import numpy as np

def check_diff(d1, d2):
    assert (np.abs(d1.astype(float)-d2.astype(float))).max()==0

class foo(object):
    # to create structure
    pass
