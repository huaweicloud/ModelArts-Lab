import mxnet as mx


def sigmoid_ce(pred, label, sample_weight=None):
    loss = mx.sym.relu(data=pred) - pred * label + \
        mx.sym.Activation(data=-mx.sym.abs(pred), act_type='softrelu')
    if sample_weight is not None:
        loss = mx.sym.broadcast_mul(loss, sample_weight)
    return mx.sym.mean(loss, axis=0, exclude=True)


def l1_loss(pred, label, sample_weight=None):
    label = mx.sym.reshape_like(label, pred)
    loss = mx.sym.abs(pred - label)
    if sample_weight is not None:
        loss = mx.sym.broadcast_mul(loss, sample_weight)
    return mx.sym.mean(loss, axis=0, exclude=True)


def get_loss(objness, box_centers, box_scales, cls_preds,
             objness_t, center_t, scale_t, weight_t, class_t, class_mask,
             denorm=10647.0, denorm_class=212940.0):
    mx_version = mx.__version__
    if mx_version >= '1.3.0':
        objness_shape = mx.sym.shape_array(data=objness_t)
        denorm = mx.sym.prod(data=mx.sym.slice_axis(
            data=objness_shape, axis=0, begin=1, end=None))
        denorm = mx.sym.cast(data=denorm, dtype='float32')
        weight_t = mx.sym.broadcast_mul(weight_t, objness_t)
        hard_objness_t = mx.sym.where(
            objness_t > 0, mx.sym.ones_like(objness_t), objness_t)
        new_objness_mask = mx.sym.where(objness_t > 0, objness_t, objness_t >= 0)
        obj_loss_ = mx.sym.broadcast_mul(
            sigmoid_ce(objness, hard_objness_t, new_objness_mask), denorm)
        center_loss_ = mx.sym.broadcast_mul(sigmoid_ce(
            box_centers, center_t, weight_t), denorm * 2)
        scale_loss_ = mx.sym.broadcast_mul(
            l1_loss(box_scales, scale_t, weight_t), denorm * 2)
        class_t_shape = mx.sym.shape_array(data=class_t)
        denorm_class = mx.sym.prod(data=mx.sym.slice_axis(
            data=class_t_shape, axis=0, begin=1, end=None))
        denorm_class = mx.sym.cast(data=denorm_class, dtype='float32')
        class_mask = mx.sym.broadcast_mul(class_mask, objness_t)
        cls_loss_ = mx.sym.broadcast_mul(sigmoid_ce(
            cls_preds, class_t, class_mask), denorm_class)
    else:
        weight_t = mx.sym.broadcast_mul(weight_t, objness_t)
        hard_objness_t = mx.sym.where(
            objness_t > 0, mx.sym.ones_like(objness_t), objness_t)
        new_objness_mask = mx.sym.where(objness_t > 0, objness_t, objness_t >= 0)
        obj_loss_ = sigmoid_ce(objness, hard_objness_t, new_objness_mask) * denorm
        center_loss_ = sigmoid_ce(box_centers, center_t, weight_t) * denorm * 2
        scale_loss_ = l1_loss(box_scales, scale_t, weight_t) * denorm * 2
        class_mask = mx.sym.broadcast_mul(class_mask, objness_t)
        cls_loss_ = sigmoid_ce(cls_preds, class_t, class_mask) * denorm_class
    total_loss_ = obj_loss_ + center_loss_ + scale_loss_ + cls_loss_
    total_loss = mx.sym.MakeLoss(data=total_loss_, name='cls_loss')
    return mx.symbol.Group([mx.symbol.BlockGrad(obj_loss_),
                            mx.symbol.BlockGrad(center_loss_),
                            mx.symbol.BlockGrad(scale_loss_),
                            mx.symbol.BlockGrad(cls_loss_),
                            total_loss])
