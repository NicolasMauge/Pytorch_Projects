import numpy as np

# CyclicLR is adapted from https://github.com/bckenstler/CLR/blob/master/clr_callback.py
def CyclicLR(base_lr=0.001, max_lr=0.006, iter_by_epoch=500, cycle_len=2, mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle', shift_phase=0):
    """This class implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            schedule_clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                iter_by_epoch=500., cycle_len=2, mode='triangular')
            lambda_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_clr)
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            schedule_clr =  = CyclicLR(base_lr=0.001, max_lr=0.006,
                                iter_by_epoch=500., cycle_len=2, scale_fn=clr_fn,
                                scale_mode='cycle')
            lambda_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=clr)
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """
    nb_iter_cycle = iter_by_epoch*cycle_len

    shift = int(nb_iter_cycle*shift_phase)

    if scale_fn is None:
        if mode == 'triangular':
            scale_fn = lambda x: 1.
            scale_mode = 'cycle'
        elif mode == 'triangular2':
            scale_fn = lambda x: 1/(2.**(x-1))
            scale_mode = 'cycle'
        elif mode == 'exp_range':
            scale_fn = lambda x: gamma**(x)
            scale_mode = 'iterations'
    
    cycle = lambda n_iter: np.floor(1+n_iter/nb_iter_cycle)
    x = lambda n_iter: np.abs(2*n_iter/nb_iter_cycle - 2*cycle(n_iter) + 1)
    if scale_mode == 'cycle':
        return lambda n_iter: base_lr + (max_lr-base_lr)*np.maximum(0, (1-x(n_iter+shift)))*scale_fn(cycle(n_iter+shift))
    else:
        return lambda n_iter: base_lr + (max_lr-base_lr)*np.maximum(0, (1-x(n_iter+shift)))*scale_fn(n_iter+shift)

def onecycle_asym(start_lr, max_lr, last_lr, epoch_max, num_epoch, iter_by_epoch):
    """
    one cycle asym
    """
    ratio_up = (max_lr-start_lr)/(iter_by_epoch*epoch_max)
    ratio_down = (max_lr-last_lr)/(iter_by_epoch*(num_epoch-epoch_max))
    def f(n_iter):
        if n_iter < epoch_max*iter_by_epoch:
            return start_lr+n_iter*ratio_up
        elif n_iter == epoch_max*iter_by_epoch:
            return max_lr
        else:
            return max_lr - (n_iter-epoch_max*iter_by_epoch)*ratio_down
    return f

def one_cycle(hp_max=0.1, epochs=10, hp_init=0.0001, hp_final=0.005, extra=5):
    """
    one cycle function: in arg = current epoch
    """
    def f(progress):
        if progress < epochs / 2:
            return 2 * hp_max * (1 - (epochs - progress) / epochs)+ hp_init
        elif progress <= epochs:
            return hp_final + 2 * (hp_max - hp_final) * (epochs - progress) / epochs
        elif progress <= epochs + extra:
            return hp_final * (extra - (progress - epochs)) / extra
        else:
            return hp_final / 10
        
    return f
    