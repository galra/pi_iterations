"""This file implements (also) two naive approaches to scanning lattices beside Z.
To find fixed points over the Z^n lattice we use the following algorithm:
    Calculate gradient descent and update x,y as usual, however, the gradient descent is calculated
    on the closest lattice point to x,y.
    If the process converges, then the closest lattice point is a fixed point of the algorithm
To find fixed points over other latices:
    We convert the new lattice to the Z lattice using the following technique:
    We supply an initial parameter to the algorithm f(a0),g(b0) instead of a0,b0,
    and an initial matrix of [df/da, 0; 0, dg/db]
"""

import basic_algo
from decimal import Decimal as dec
import numpy as np
import itertools

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class GradientDescentBasicAlgo:
    def __init__(self, enforce_Z=False, f_x=None, dfdx=None, g_y=None, dgdy=None,
                 max_num_of_iterations = 50000, threshold=None, step_size='dynamic'):
        # lattice parameters
        self.enforce_Z = enforce_Z
        if f_x is None:
            f_x = lambda x: x
        if g_y is None:
            g_y = f_x
        if dfdx is None:
            dfdx = lambda x: 1
        if dgdy is None:
            dgdy = dfdx
        self.f, self.g, self.dfdx, self.dgdy = f_x, g_y, dfdx, dgdy

        # Grad-Des parameters
        if threshold is None:
            threshold = dec(10)**dec(-2)
        self.threshold = threshold
        if step_size is None:
            step_size = [dec(0.001)] * 2
        self.step_size = step_size
        self.max_num_of_iterations = max_num_of_iterations

        # initial values parameters
        self.default_pi0 = dec(2)+dec(2).sqrt()
        self.default_x1 = dec(0.5)*(dec(2).sqrt().sqrt() + dec(2)**dec('-0.25'))
        self.default_y1 = dec(2).sqrt().sqrt()

        # set decimal precision
        basic_algo.set_precision(100)

    def find_params(self, x1=None, y1=None, pi0=None, show_progress=True):
        """Runs gradient-descent with the given parameters. step-size can be either dynamic of a number.
        dynamic is good"""
        # init initial values
        if pi0 is None:
            pi0 = self.default_pi0
        if x1 is None:
            x1 = self.default_x1
        if y1 is None:
            y1 = self.default_y1
        if self.step_size != 'dynamic':
            dxdy = self.step_size
        if isinstance(x1, int) or isinstance(x1, float):
            x1 = dec(x1)
        if isinstance(y1, int) or isinstance(y1, float):
            y1 = dec(y1)
        if isinstance(pi0, int) or isinstance(pi0, float):
            pi0 = dec(pi0)

        xy = np.array((x1, y1))
        if self.enforce_Z:
            x_param = dec(x1.__round__())
            y_param = dec(y1.__round__())
        else:
            x_param = x1
            y_param = y1

        processed_x, processed_y, first_diff_mat = self.z_to_arbitrary_lattice(x_param, y_param,
                                                                               self.f, self.dfdx, self.g, self.dgdy)
        ba = basic_algo.PiBasicAlgo(x1=processed_x, y1=processed_y, pi0=pi0, first_diff_mat=first_diff_mat)
        ba.gen_iterations(10)
        iter_num = 1
        dec_0 = dec(0)

        while (abs(ba.compare_result()) > self.threshold) and (iter_num < self.max_num_of_iterations):
            grad = ba.get_derivative()
            if self.step_size == 'dynamic':
                dxdy = dec(0.0001) * np.array((abs(grad[0,0]).log10().max(dec(1)),
                                                  abs(grad[0,1]).log10().max(dec(1))))
            xy_diff = np.multiply(-grad, dxdy)
            xy = xy + xy_diff
            x_param, y_param = xy.tolist()[0]
            if x_param <= dec_0 or y_param <= dec_0:
                break
            if self.enforce_Z:
                x_param = dec(x_param.__round__())
                y_param = dec(y_param.__round__())
            processed_x, processed_y, first_diff_mat = self.z_to_arbitrary_lattice(x_param, y_param,
                                                                               self.f, self.dfdx, self.g, self.dgdy)
            ba.reinitialize(processed_x, processed_y, first_diff_mat=first_diff_mat)
            ba.gen_iterations(10)
            iter_num += 1
            if iter_num % 1000 == 0 and show_progress:
                print('\r%d' % iter_num, end='')
        print('')

        if iter_num >= self.max_num_of_iterations and show_progress:
            print('Iterations limit reached. Aborting.')
        elif show_progress and (x_param <= dec_0 or y_param <= dec_0):
            print('Parameters reached non-positive value. Aborting.')
        elif show_progress:
            print('Result distance: %s' % abs(ba.compare_result()))

        if iter_num >= self.max_num_of_iterations or x_param <= dec_0 or y_param <= dec_0:
            return
        return xy.tolist()[0]

    def z_to_arbitrary_lattice(self, x1, y1, f, dfdx, g='same', dgdy='same'):
        if g == 'same':
            g = f
        if dgdy == 'same':
            dgdy = dfdx
        first_diff_mat = np.matrix(((dec(dfdx(x1)), dec(0)), (dec(0), dec(dgdy(y1)))), dtype=dec)
        f_x1 = f(x1)
        g_y1 = g(y1)
        return f_x1, g_y1, first_diff_mat

def gen_pi_map(x1_range, y1_range, pi0=None, resolution=100, iterations=10):
    """Plots a map of the function iterations results over a square range"""
    map_data = {'x1': [], 'y1': [], 'pi': []}
    if pi0 is None:
        pi0 = dec(2) + dec(2).sqrt()
    ba = basic_algo.PiBasicAlgo(x1=dec(1), y1=dec(1), pi0=dec(1))

    if x1_range[0] < 0 or y1_range[0] < 0:
        raise ValueError('Range most be of positive values!')

    if x1_range[0] >= x1_range[1] or y1_range[0] >= y1_range[1]:
        raise ValueError('Range boundaries are disordered!')

    x_space = np.linspace(x1_range[0], x1_range[1], resolution)
    y_space = np.linspace(y1_range[0], y1_range[1], resolution)
    map_data['x1'], map_data['y1'] = np.meshgrid(x_space, y_space)
    map_data['pi'] = np.array(map_data['x1'])
    for x1_i in range(len(x_space)):
        for y1_i in range(len(y_space)):
            ba.reinitialize(dec(x_space[x1_i]), dec(y_space[y1_i]), dec(pi0))
            ba.gen_iterations(iterations)
            map_data['pi'][x1_i,y1_i] = (ba.compare_result()**dec(2)).log10()

    return map_data

def gen_pi_path(x1, y1, pi0=None, iterations=10):
    """Calculates the whole path of x,y,pi during the iterations process
    :param x1:
    :param y1:
    :param pi0:
    :param iterations:
    :return:
    """
    if pi0 is None:
        pi0 = dec(2) + dec(2).sqrt()
    ba = basic_algo.PiBasicAlgo(x1=x1, y1=y1, pi0=pi0)
    ba.gen_iterations(iterations)
    return {'x': ba.xs, 'y': ba.ys, 'pi': ba.pis}

def draw_map(map_data, fig):
    """Plots a map to a given figure"""
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(map_data['x1'], map_data['y1'], map_data['pi']) #, rstride=1, cstride=1)

def draw_path(path_data, fig):
    """Plots a given path of x,y,pi (path_data is a dictionary) to a given figure"""
    ax = fig.gca(projection='3d')
    ax.plot(path_data['x'], path_data['y'], path_data['pi']) #, label='parametric curve')

def draw_all(x_range=[0.1, 2], y_range=[0.1, 2], starting_points=None,
             pi0=None, resolution=100, iterations=10, zero_threshold=0.001):
    if pi0 is None:
        pi0 = dec(2) + dec(2).sqrt()
    if starting_points is None:
        x_space = np.linspace(x_range[0], x_range[1], resolution)
        y_space = np.linspace(y_range[0], y_range[1], resolution)
        starting_points = itertools.product(x_space, y_space)

    fig = plt.figure()
    map_data = gen_pi_map(x_range, y_range, pi0, resolution, iterations)
    draw_map(map_data, fig)

    # find the path coordinates
    path_xy = [ (x_space[i], y_space[j])
                for i,j in itertools.product(range(len(x_space)), range(len(y_space)))
                if map_data['pi'][i,j] < -3.5 ]

    # zero_path = [ (x_space[i], y_space[j], map_data['pi'][i,j])
    #               for i,j in i,j in itertools(range(len(x_space)), range(len(y_space)))
    #               if map_data['pi'][i,j]) < zero_threshold ]
    path_x, path_y = [ x for x,y in path_xy ], [ y for x,y in path_xy ]
    fig_path = plt.figure()
    ax_path = fig_path.gca()
    ax_path.plot(path_x, path_y, 'b.')

    fig.show()
    fig_path.show()
    return (path_x, path_y)

def array_to_matlab(array_name, array):
    """Converts an array a to string to paste to matlab"""
    return '%s = [%s];' % (array_name, ' '.join([ i.__repr__() for i in array ]))

