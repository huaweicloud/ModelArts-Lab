import sys
import os

pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(pwd + ""))
print("load path %s" % sys.path[0])

from .config import *
