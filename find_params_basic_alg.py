import basic_algo
from decimal import Decimal as dec
import numpy as np
import itertools

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def find_params_basic_alg(x1=None, y1=None, pi0=None, step_size='dynamic', threshold=None):
    basic_algo.set_precision(100)
    if pi0 is None:
        pi0 = dec(2)+dec(2).sqrt()
    if x1 is None:
        x1 = dec(0.5)*(dec(2).sqrt().sqrt() + dec(2)**dec('-0.25'))
    if y1 is None:
        y1 = dec(2).sqrt().sqrt()
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

def gen_pi_map(x1_range, y1_range, pi0=None, resolution=100, iterations=10):
    map_data = {'x1': [], 'y1': [], 'pi': []}
    if pi0 is None:
        pi0 = dec(2) + dec(2).sqrt()
    ba = basic_algo.PiBasicAlgo(x1=dec(1), y1=dec(1), pi0=dec(1))

    if x1_range[0] < 0 or y1_range[0] < 0:
        raise ValueError('Range most be of positive values!')

    if x1_range[0] >= x1_range[1] or y1_range[0] >= y1_range[1]:
        raise ValueError('Range boundaries are disordered!')

    for x1 in np.linspace(x1_range[0], x1_range[1], resolution):
        for y1 in np.linspace(y1_range[0], y1_range[1], resolution):
            if x1 == 0 or y1 == 0:
                continue
            ba.reinitialize(dec(x1), dec(y1), dec(pi0))
            map_data['x1'].append(x1)
            map_data['y1'].append(y1)
            ba.gen_iterations(iterations)
            map_data['pi'].append(ba.pis[-1])

    return map_data

def gen_pi_path(x1, y1, pi0=None, iterations=10):
    if pi0 is None:
        pi0 = dec(2) + dec(2).sqrt()
    ba = basic_algo.PiBasicAlgo(x1=x1, y1=y1, pi0=pi0)
    ba.gen_iterations(iterations)
    return {'x': ba.xs, 'y': ba.ys, 'pi': ba.pis}

def draw_map(map_data, fig):
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(map_data['x1'], map_data['y1'], map_data['pi']) #, rstride=1, cstride=1)

def draw_path(path_data, fig):
    ax = fig.gca(projection='3d')
    ax.plot(path_data['x'], path_data['y'], path_data['pi']) #, label='parametric curve')

def draw_all(x_range=[1, 10], y_range=[1, 10], starting_points=None,
             pi0=None, resolution=10, iterations=10):
    if pi0 is None:
        pi0 = dec(2) + dec(2).sqrt()
    if starting_points is None:
        starting_points = itertools.product(range(*x_range), range(*y_range))

    fig = plt.figure()
    map_data = gen_pi_map(x_range, y_range, pi0, resolution, iterations)
    draw_map(map_data, fig)

    fig.show()




