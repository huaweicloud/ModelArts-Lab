import os
import os.path as osp

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
FLAGS2 = {}

######################
# General Parameters #
######################
FLAGS2["pixel_means"] = np.array([[[102.9801, 115.9465, 122.7717]]])
tf.app.flags.DEFINE_integer('rng_seed', 3, "Tensorflow seed for reproducibility")

tf.app.flags.DEFINE_string('pooling_mode', "crop", "Default pooling mode, only 'crop' is available")

######################
# Network Parameters #
######################
tf.app.flags.DEFINE_string('network', "resnet_v1", "The network to be used as backbone")

#######################
# Training Parameters #
#######################
tf.app.flags.DEFINE_float('weight_decay', 0.0005, "Weight decay, for regularization")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "Learning rate")
tf.app.flags.DEFINE_float('momentum', 0.9, "Momentum")
tf.app.flags.DEFINE_float('gamma', 0.1, "Factor for reducing the learning rate")

tf.app.flags.DEFINE_integer('batch_size', 4, "Network batch size during training")
tf.app.flags.DEFINE_integer('max_iters', 20000, "Max iteration")
tf.app.flags.DEFINE_integer('step_size', 30000, "Step size for reducing the learning rate, currently only support one step")
tf.app.flags.DEFINE_integer('display', 10, "Iteration intervals for showing the loss during training, on command line interface")

tf.app.flags.DEFINE_integer('fixed_blocks', 1, "Number of fixed blocks during training, by default the first of all 4 blocks is fixed")
tf.app.flags.DEFINE_string('initializer', "truncated", "Network initialization parameters")
# tf.app.flags.DEFINE_string('pretrained_model', "./data/imagenet_weights/vgg16.ckpt", "Pretrained network weights")
tf.app.flags.DEFINE_string('pretrained_model', "./data/imagenet_weights/resnet_v1_50.ckpt", "Pretrained network weights")

tf.app.flags.DEFINE_boolean('bias_decay', False, "Whether to have weight decay on bias as well")
tf.app.flags.DEFINE_boolean('double_bias', True, "Whether to double the learning rate for bias")
tf.app.flags.DEFINE_boolean('use_all_gt', True, "Whether to use all ground truth bounding boxes for training, "
                                                "For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''")

tf.app.flags.DEFINE_boolean('max_pool', False, "resized to a square of POOLING_SIZE")

tf.app.flags.DEFINE_integer('max_size', 1000, "Max pixel size of the longest side of a scaled input image")
tf.app.flags.DEFINE_integer('test_max_size', 1000, "Max pixel size of the longest side of a scaled input image")
tf.app.flags.DEFINE_integer('ims_per_batch', 1, "Images to use per minibatch")
tf.app.flags.DEFINE_integer('snapshot_iterations', 5000, "Iteration to take snapshot")

FLAGS2["scales"] = (600,)
FLAGS2["test_scales"] = (600,)

######################
# Testing Parameters #
######################
tf.app.flags.DEFINE_string('test_mode', "top", "Test mode for bbox proposal")  # nms, top

##################
# RPN Parameters #
##################
tf.app.flags.DEFINE_float('rpn_negative_overlap', 0.3, "IOU < thresh: negative example")
tf.app.flags.DEFINE_float('rpn_positive_overlap', 0.7, "IOU >= thresh: positive example")
tf.app.flags.DEFINE_float('rpn_fg_fraction', 0.5, "Max number of foreground examples")
tf.app.flags.DEFINE_float('rpn_train_nms_thresh', 0.7, "NMS threshold used on RPN proposals")
tf.app.flags.DEFINE_float('rpn_test_nms_thresh', 0.7, "NMS threshold used on RPN proposals")

tf.app.flags.DEFINE_integer('rpn_train_pre_nms_top_n', 12000, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_train_post_nms_top_n', 2000, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_test_pre_nms_top_n', 6000, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_test_post_nms_top_n', 300, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_batchsize', 256, "Total number of examples")
tf.app.flags.DEFINE_integer('rpn_positive_weight', -1,
                            'Give the positive RPN examples weight of p * 1 / {num positives} and give negatives a weight of (1 - p).'
                            'Set to -1.0 to use uniform example weighting')
tf.app.flags.DEFINE_integer('rpn_top_n', 300, "Only useful when TEST.MODE is 'top', specifies the number of top proposals to select")

tf.app.flags.DEFINE_boolean('rpn_clobber_positives', False, "If an anchor satisfied by positive and negative conditions set to negative")

#######################
# Proposal Parameters #
#######################
tf.app.flags.DEFINE_float('proposal_fg_fraction', 0.25, "Fraction of minibatch that is labeled foreground (i.e. class > 0)")
tf.app.flags.DEFINE_boolean('proposal_use_gt', False, "Whether to add ground truth boxes to the pool when sampling regions")

###########################
# Bounding Box Parameters #
###########################
tf.app.flags.DEFINE_float('roi_fg_threshold', 0.5, "Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)")
tf.app.flags.DEFINE_float('roi_bg_threshold_high', 0.5, "Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))")
tf.app.flags.DEFINE_float('roi_bg_threshold_low', 0.1, "Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))")

tf.app.flags.DEFINE_boolean('bbox_normalize_targets_precomputed', True, "# Normalize the targets using 'precomputed' (or made up) means and stdevs (BBOX_NORMALIZE_TARGETS must also be True)")
tf.app.flags.DEFINE_boolean('test_bbox_reg', True, "Test using bounding-box regressors")

FLAGS2["bbox_inside_weights"] = (1.0, 1.0, 1.0, 1.0)
FLAGS2["bbox_normalize_means"] = (0.0, 0.0, 0.0, 0.0)
FLAGS2["bbox_normalize_stds"] = (0.1, 0.1, 0.1, 0.1)

##################
# ROI Parameters #
##################
tf.app.flags.DEFINE_integer('roi_pooling_size', 7, "Size of the pooled region after RoI pooling")

######################
# Dataset Parameters #
######################
FLAGS2["root_dir"] = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
FLAGS2["data_dir"] = osp.abspath(osp.join(FLAGS2["root_dir"], 'data'))


def get_output_dir(imdb, weights_filename):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(FLAGS2["root_dir"], FLAGS2["root_dir"] , 'default', imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir
