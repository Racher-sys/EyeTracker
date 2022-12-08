import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import PIL.Image as Image
import os
imagefile = "image/aoi&target&ped/zjb2/0.png"
img = Image.open(imagefile)
print(img.size)
cropped = img.crop((0, 0, 200, 30))  # (left, upper, right, lower)
cropped.save("test.png")