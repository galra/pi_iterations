import numpy as np
from decimal import Decimal as dec
import decimal
import gen_real_pi

def set_precision(prec):
    decimal.getcontext().prec=prec

class PiRatioIterations:
    def __init__(self, x1, y1, pi0, diff_mat_gen, first_diff_mat, dtype=dec):
        self.x1 = x1
        self.y1 = y1
        self.pi0 = pi0
        self.dtype = dtype
        self.diff_mat_gen = diff_mat_gen
        self.first_diff_mat = first_diff_mat
        self.diff_vectors = []

    def reinitialize(self, x1, y1, pi0, first_diff_mat=None):
        self.x1 = x1
        self.y1 = y1
        self.pi0 = pi0
        if first_diff_mat is not None:
            self.first_diff_mat = first_diff_mat

    def gen_iterations(self, num_of_iters):
        raise NotImplemented()


class PiBasicAlgo(PiRatioIterations):
    def __init__(self, x1=None, y1=None, pi0=None,
                 diff_mat_gen=None, first_diff_mat=None, dtype=dec):
        if not x1:
            x1 = dec(2).sqrt()
            x1 = (x1.sqrt() + dec(1) / x1.sqrt()) / dec(2)
        if not y1:
            y1 = dec(2).sqrt().sqrt()
        if not pi0:
            pi0 = (dec(2)+dec(2).sqrt())
        if not diff_mat_gen:
            dec_1 = dec(1)
            dec_2 = dec(2)
            dec_half = dec(0.5)
            dec_quarter = dec(0.25)
            dec_min_3_2 = dec(-1.5)
            diff_mat_gen = lambda x, y: np.matrix(((dec_quarter * (dec_1/x.sqrt() - x**dec_min_3_2), 0),
                                                   (dec_half/(y+dec_1) * (dec_1/x.sqrt() - x**dec_min_3_2),
                                                    x.sqrt()/(y + dec_1) -
                                                    (y * x.sqrt() + dec_1/x.sqrt()) / (y + dec_1)**dec_2)), dtype=dtype)
        if first_diff_mat is None:
            first_diff_mat = np.matrix(((dec(1), dec(0)), (dec(0), dec(1))), dtype=dtype)
        PiRatioIterations.__init__(self, x1, y1, pi0, diff_mat_gen, first_diff_mat, dtype)

    def reinitialize(self, x1=None, y1=None, pi0=None, first_diff_mat=None):
        if not x1:
            x1 = dec(2).sqrt()
            x1 = (x1.sqrt() + dec(1) / x1.sqrt()) / dec(2)
        if not y1:
            y1 = dec(2).sqrt().sqrt()
        if not pi0:
            pi0 = (dec(2)+dec(2).sqrt())
        PiRatioIterations.reinitialize(self, x1, y1, pi0, first_diff_mat)

    def gen_iterations(self, num_of_iters):
        self.xs = [self.x1]
        self.ys = [self.y1]
        self.pis = [self.pi0]
        self.diff_matrices = [self.first_diff_mat]
        self.diff_vectors = []
        append_x = self.xs.append
        append_y = self.ys.append
        append_pi = self.pis.append
        append_diff_mat = self.diff_matrices.append
        append_diff_vec = self.diff_vectors.append


        dec_1 = dec(1)
        dec_2 = dec(2)
        x = self.x1
        y = self.y1
        pi = self.pi0
        diff_mat = self.first_diff_mat
        diff_vectors_sums = np.matrix((dec(0), dec(0)))
        for i in range(num_of_iters):
            x_sqrt = x.sqrt()
            x_inv_sqrt = dec_1 / x_sqrt
            x, y, pi = ((x_sqrt + x_inv_sqrt) / dec_2,
                        (y * x_sqrt + x_inv_sqrt) / (y + dec_1),
                        pi * (x + dec_1) / (y + dec_1))
            diff_vectors_sums = diff_vectors_sums + np.matrix((dec_1/x, -dec_1/y)) * diff_mat
            diff_mat = self.diff_mat_gen(x, y) * diff_mat
            append_x(x)
            append_y(y)
            append_pi(pi)
            append_diff_vec(pi*diff_vectors_sums)
            append_diff_mat(diff_mat)

    def compare_result(self, real_pi=None):
        if not real_pi:
            real_pi = gen_real_pi.gen_real_pi()
        try:
            return real_pi - self.pis[-1]
        except:
            raise RuntimeError('Run gen_iterations first. No PI was generated!')

    def get_derivative(self):
        return -self.compare_result() * self.diff_vectors[-1]





