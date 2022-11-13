import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import PIL.Image as Image
import os

sub = [[1, 2, 3], [4, 5, 6], [2, 3, 8]]
sub2 = [[1, 2, 3], [4, 5, 6]]
sub3 = [[1, 2, 3], [4, 5, 6]]

a = np.max(sub, axis=0)

sub1 = np.vstack((sub, sub2, sub3))
print(a)
