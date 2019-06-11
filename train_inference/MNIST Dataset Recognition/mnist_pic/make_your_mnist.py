# -*- coding: utf-8 -*-

"""
Make mnist data set.
Ensure that the PIL module is installed before using the code below.
"""
from PIL import Image
cache = '/your/path/'
# source pic
sr = cache + '0.jpg'
# target pic
tg = cache + '00.jpg'
I = Image.open(sr)
# resize to 28 * 28
I = I.resize((28, 28))
# translating a color image to black and white 
L = I.convert('L')
L.save(tg)
