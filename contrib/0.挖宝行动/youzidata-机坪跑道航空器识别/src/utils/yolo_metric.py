import numpy as np

import mxnet as mx


class YoloLoss(mx.metric.EvalMetric):
    """Dummy metric for directly printing yolo loss.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    index : int
        Use the idxth preds to compute loss
    """
    def __init__(self, name='loss',
                 output_names=None, label_names=None, index=-1):
        super(YoloLoss, self).__init__(
            name, output_names=output_names, label_names=label_names)
        self._pred_idx = index

    def update(self, _, preds):
        if self._pred_idx > 0:
            preds = preds[self._pred_idx]
        if isinstance(preds,  mx.nd.NDArray):
            preds = [preds]
        for pred in preds:
            self.sum_metric += mx.nd.sum(pred).asscalar()
            self.num_inst += pred.size


class YoloVOCMApMetric(mx.metric.EvalMetric):
    """
    Calculate mean AP for object detection task

    Parameters:
    ---------
    iou_thresh : float
        IOU overlap threshold for TP
    class_names : list of str
        optional, if provided, will print out AP for each class
    """

    def __init__(self, iou_thresh=0.5, class_names=None):
        super(YoloVOCMApMetric, self).__init__('VOCMeanAP')
        if class_names is None:
            self.num = None
        else:
            assert isinstance(class_names, (list, tuple))
            for name in class_names:
                assert isinstance(name, str), "must provide names as str"
            num = len(class_names)
            self.name = list(class_names) + ['mAP']
            self.num = num + 1
        self.reset()
        self.iou_thresh = iou_thresh
        self.class_names = class_names

    def reset(self):
        """Clear the internal statistics to initial state."""
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num
        from collections import defaultdict
        self._n_pos = defaultdict(int)
        self._score = defaultdict(list)
        self._match = defaultdict(list)

    def get(self):
        """Get the current evaluation result.

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        self._update()  # update metric at this time
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s' % (self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)

    # pylint: disable=arguments-differ, too-many-nested-blocks
    def update(self,
               pred_bboxes,
               pred_labels,
               pred_scores,
               gt_bboxes,
               gt_labels,
               gt_difficults=None):
        """Update internal buffer with latest prediction and gt pairs.

        Parameters
        ----------
        pred_bboxes : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes with shape `B, N, 4`.
            Where B is the size of mini-batch, N is the number of bboxes.
        pred_labels : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes labels with shape `B, N`.
        pred_scores : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes scores with shape `B, N`.
        gt_bboxes : mxnet.NDArray or numpy.ndarray
            Ground-truth bounding boxes with shape `B, M, 4`.
            Where B is the size of mini-batch, M is the number of ground-truths.
        gt_labels : mxnet.NDArray or numpy.ndarray
            Ground-truth bounding boxes labels with shape `B, M`.
        gt_difficults : mxnet.NDArray or numpy.ndarray, optional, default is None
            Ground-truth bounding boxes difficulty labels with shape `B, M`.

        """

        def as_numpy(a):
            """Convert a (list of) mx.NDArray into numpy.ndarray"""
            if isinstance(a, (list, tuple)):
                out = [
                    x.asnumpy() if isinstance(x, mx.nd.NDArray) else x
                    for x in a
                ]
                try:
                    out = np.concatenate(out, axis=0)
                except ValueError:
                    out = np.array(out)
                return out
            elif isinstance(a, mx.nd.NDArray):
                a = a.asnumpy()
            return a

        if gt_difficults is None:
            gt_difficults = [None for _ in gt_labels]

        for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in zip(
                *[
                    as_numpy(x) for x in [
                        pred_bboxes, pred_labels, pred_scores, gt_bboxes,
                        gt_labels, gt_difficults
                    ]
                ]):
            # strip padding -1 for pred and gt
            valid_pred = np.where(pred_label.flat >= 0)[0]
            pred_bbox = pred_bbox[valid_pred, :]
            pred_label = pred_label.flat[valid_pred].astype(int)
            pred_score = pred_score.flat[valid_pred]
            valid_gt = np.where(gt_label.flat >= 0)[0]
            gt_bbox = gt_bbox[valid_gt, :]
            gt_label = gt_label.flat[valid_gt].astype(int)
            if gt_difficult is None:
                gt_difficult = np.zeros(gt_bbox.shape[0])
            else:
                gt_difficult = gt_difficult.flat[valid_gt]

            for l in np.unique(
                    np.concatenate((pred_label, gt_label)).astype(int)):
                pred_mask_l = pred_label == l
                pred_bbox_l = pred_bbox[pred_mask_l]
                pred_score_l = pred_score[pred_mask_l]
                # sort by score
                order = pred_score_l.argsort()[::-1]
                pred_bbox_l = pred_bbox_l[order]
                pred_score_l = pred_score_l[order]

                gt_mask_l = gt_label == l
                gt_bbox_l = gt_bbox[gt_mask_l]
                gt_difficult_l = gt_difficult[gt_mask_l]

                self._n_pos[l] += np.logical_not(gt_difficult_l).sum()
                self._score[l].extend(pred_score_l)

                if len(pred_bbox_l) == 0:
                    continue
                if len(gt_bbox_l) == 0:
                    self._match[l].extend((0, ) * pred_bbox_l.shape[0])
                    continue

                # VOC evaluation follows integer typed bounding boxes.
                pred_bbox_l = pred_bbox_l.copy()
                pred_bbox_l[:, 2:] += 1
                gt_bbox_l = gt_bbox_l.copy()
                gt_bbox_l[:, 2:] += 1

                iou = self._bbox_iou(pred_bbox_l, gt_bbox_l)
                gt_index = iou.argmax(axis=1)
                # set -1 if there is no matching ground truth
                gt_index[iou.max(axis=1) < self.iou_thresh] = -1
                del iou

                selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
                for gt_idx in gt_index:
                    if gt_idx >= 0:
                        if gt_difficult_l[gt_idx]:
                            self._match[l].append(-1)
                        else:
                            if not selec[gt_idx]:
                                self._match[l].append(1)
                            else:
                                self._match[l].append(0)
                        selec[gt_idx] = True
                    else:
                        self._match[l].append(0)

    def _update(self):
        """ update num_inst and sum_metric """
        aps = []
        rec_prec_ap = {}
        class_metrics_map = {}
        rec_avg = 0
        prec_avg = 0
        ap_avgs = 0
        rec_avgs = 0
        prec_avgs = 0
        recall, precs = self._recall_prec()
        for l, rec, prec in zip(range(len(precs)), recall, precs):
            ap = self._average_precision(rec, prec)
            aps.append(ap)
            rec_avg = rec.sum() / rec.size
            rec_avgs += rec_avg
            prec_avg = prec.sum() / prec.size
            prec_avgs += prec_avg
            rec_prec_ap['id'] = l
            rec_prec_ap['recall'] = rec_avg
            rec_prec_ap['precision'] = prec_avg
            rec_prec_ap['accuracy'] = ap
            ap_avgs += ap
            class_metrics_map[l] = rec_prec_ap.copy()
            if self.num is not None and l < (self.num - 1):
                self.sum_metric[l] = ap
                self.num_inst[l] = 1
        if self.num is None:
            self.num_inst = 1
            self.sum_metric = np.nanmean(aps)
        else:
            self.num_inst[-1] = 1
            self.sum_metric[-1] = np.nanmean(aps)
        rec_prec_ap['recall'] = rec_avgs / (self.num - 1)
        rec_prec_ap['precision'] = prec_avgs / (self.num - 1)
        rec_prec_ap['accuracy'] = ap_avgs / (self.num - 1)
        class_metrics_map['total'] = rec_prec_ap

    def _recall_prec(self):
        """ get recall and precision from internal records """
        n_fg_class = max(self._n_pos.keys()) + 1
        prec = [None] * n_fg_class
        rec = [None] * n_fg_class

        for l in self._n_pos.keys():
            score_l = np.array(self._score[l])
            match_l = np.array(self._match[l], dtype=np.int32)

            order = score_l.argsort()[::-1]
            match_l = match_l[order]

            tp = np.cumsum(match_l == 1)
            fp = np.cumsum(match_l == 0)

            # If an element of fp + tp is 0,
            # the corresponding element of prec[l] is nan.
            with np.errstate(divide='ignore', invalid='ignore'):
                prec[l] = tp.astype('float32') / (fp + tp)
            # If n_pos[l] is 0, rec[l] is None.
            if self._n_pos[l] > 0:
                rec[l] = tp.astype('float32') / self._n_pos[l]

        return rec, prec

    def _average_precision(self, rec, prec):
        """
        calculate average precision, override the default one,
        special 11-point metric

        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        if rec is None or prec is None:
            return np.nan
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(np.nan_to_num(prec)[rec >= t])
            ap += p / 11.
        return ap

    def _bbox_iou(self, bbox_a, bbox_b, offset=0):
        """Calculate Intersection-Over-Union(IOU) of two bounding boxes.

        Parameters
        ----------
        bbox_a : numpy.ndarray
            An ndarray with shape :math:`(N, 4)`.
        bbox_b : numpy.ndarray
            An ndarray with shape :math:`(M, 4)`.
        offset : float or int, default is 0
            The ``offset`` is used to control the whether the width(or height) is computed as
            (right - left + ``offset``).
            Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray
            An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
            bounding boxes in `bbox_a` and `bbox_b`.

        """
        if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
            raise IndexError(
                "Bounding boxes axis 1 must have at least length 4")

        tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
        br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

        area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
        area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
        area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
        return area_i / (area_a[:, None] + area_b - area_i)
