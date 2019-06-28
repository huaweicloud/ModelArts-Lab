import os
import sys

root = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, root)
sys.path.insert(0, "%s/lib" % root)
sys.path.insert(0, "%s/core" % root)
sys.path.insert(0, "%s/cloud" % root)
sys.path.insert(0, "%s/config" % root)

from dog_and_cat_train import Program

if __name__ == "__main__":
    sys.exit(Program(sys.argv[1:]))

