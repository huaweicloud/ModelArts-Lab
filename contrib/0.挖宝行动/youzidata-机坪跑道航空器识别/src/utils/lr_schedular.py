# Copyright 2018 Deep Learning Service of Huawei Cloud. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import logging
from math import cos, pi


class WarmUpLRScheduler(object):
    """Base class of a learning rate scheduler.
    A scheduler returns a new learning rate based on the number of updates that have
    been performed.
    Parameters
    ----------
    base_lr : float, optional
        The initial learning rate.
    """
    def __init__(self, base_lr=0.01, warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):

        self.base_lr = base_lr

        assert isinstance(warmup_steps, int)
        self.warmup_steps = warmup_steps
        self.warmup_final_lr = base_lr
        self.warmup_begin_lr = warmup_begin_lr

        if warmup_steps > 0:
            if self.warmup_begin_lr > self.warmup_final_lr:
                raise ValueError("Base lr has to be higher than warmup_begin_lr")
            if self.warmup_steps < 0:
                raise ValueError("Warmup steps has to be positive or 0")
        if warmup_mode not in ['linear', 'constant']:
            raise ValueError("Supports only linear and constant modes of warmup")
        self.warmup_mode = warmup_mode

    def get_warmup_lr(self, num_update):
        assert num_update <= self.warmup_steps
        if self.warmup_mode == 'linear':
            if self.warmup_steps > 0:
                increase = (self.warmup_final_lr - self.warmup_begin_lr) \
                           * float(num_update)/float(self.warmup_steps)
                return self.warmup_begin_lr + increase
            else:
                return self.base_lr
        elif self.warmup_mode == 'constant':
            return self.warmup_begin_lr
        else:
            raise ValueError("Invalid warmup mode %s", self.warmup_mode)

    def __call__(self, num_update):
        """Return a new learning rate.
        The ``num_update`` is the upper bound of the number of updates applied to
        every weight.
        Assume the optimizer has updated *i*-th weight by *k_i* times, namely
        ``optimizer.update(i, weight_i)`` is called by *k_i* times. Then::
            num_update = max([k_i for all i])
        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        raise NotImplementedError("must override this")


class WarmUpMultiFactorScheduler(WarmUpLRScheduler):
    """Reduce the learning rate by given a list of steps.
    Assume there exists *k* such that::
       step[k] <= num_update and num_update < step[k+1]
    Then calculate the new learning rate by::
       base_lr * pow(factor, k+1)
    Parameters
    ----------
    step: list of int
        The list of steps to schedule a change
    factor: float
        The factor to change the learning rate.
    """
    def __init__(self, step, factor=1, base_lr=0.01, warmup_steps=0, warmup_begin_lr=0,
                 warmup_mode='linear'):
        super(WarmUpMultiFactorScheduler, self).__init__(base_lr, warmup_steps,
                                                   warmup_begin_lr, warmup_mode)
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0

    def __call__(self, num_update):
        if num_update <= self.warmup_steps:
            return self.get_warmup_lr(num_update)

        # NOTE: use while rather than if  (for continuing training via load_epoch)
        while self.cur_step_ind <= len(self.step)-1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= self.factor
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
            else:
                return self.base_lr
        return self.base_lr
