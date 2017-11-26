import basic_algo
from decimal import Decimal as dec
import numpy as np

def find_params_basic_alg(x1=None, y1=None, pi0=None, step_size='dynamic', threshold=None):
    basic_algo.set_precision(100)
    if pi0 is None:
        pi0 = dec(2)+dec(2).sqrt()
    if step_size is None:
        dxdy = [dec(0.001)] * 2
    elif step_size != 'dynamic':
        dxdy = step_size
    if threshold is None:
        threshold = dec(10)**dec(-2)
    if isinstance(x1, int) or isinstance(x1, float):
        x1 = dec(x1)
    if isinstance(y1, int) or isinstance(y1, float):
        y1 = dec(y1)
    if isinstance(pi0, int) or isinstance(pi0, float):
        pi0 = dec(pi0)
    ba = basic_algo.PiBasicAlgo(x1=x1, y1=y1)
    ba.gen_iterations(10)
    xy = np.array((x1, y1))
    while abs(ba.compare_result()) > threshold:
        grad = ba.get_derivative()
        if step_size == 'dynamic':
            dxdy = dec(0.0001) * np.array((abs(grad[0,0]).log10().max(dec(1)),
                                              abs(grad[0,1]).log10().max(dec(1))))
        xy_diff = np.multiply(-grad, dxdy)
        xy = xy + xy_diff
        ba.reinitialize(*(xy.tolist()[0]))
        ba.gen_iterations(10)

    return xy.tolist()


