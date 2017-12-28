import itertools
import find_params_basic_alg
import math
from decimal import Decimal as dec
from multiprocessing import Pool

def latice_exploration_core_task(funcs_chunck):
    import find_params_basic_alg
    exps = [ {'f_x': lambda x: n**x, 'dfdx': lambda x: n.ln() * n**x}
             for n in [ dec(i) for i,t in funcs_chunck if t == 'exps' ] ]
    roots = [ {'f_x': lambda x: x**q, 'dfdx': lambda x: q * x**(q-dec(1))}
               for q in [ dec(1)/dec(i) for i,t in funcs_chunck if t == 'roots' ] ]
    funcs = exps + roots
    results = []
    b_a = lambda a: 0.9878*a**0.8285
    for f_dict in funcs:
        gd = find_params_basic_alg.GradientDescentBasicAlgo(enforce_Z=True, **f_dict)
        for x in range(2,5):
            basic_y = round(b_a(x))
            for y in range(max(basic_y-2, 1), basic_y+2):
                res = gd.find_params(x, y, show_progress=False)
                if res:
                    results.append({'values': res, 'f_dict': f_dict})
    return results

def latices_exploration(cores=8):
    exps = [ (n,'exps') for n in [ dec(i) for i in range(2,4) ] ]
    roots = [ (q, 'roots') for q in [ dec(1)/dec(i) for i in range(1,4) ] ]
    # exps = [ {'f': lambda x: n**x, 'dfdx': lambda x: n.ln() * n**x} for n in [ dec(i) for i in range(2,4) ] ]
    # roots = [ {'f': lambda x: x**q, 'dfdx': lambda x: q * x**(q-1)} for q in [ dec(1)/dec(i) for i in range(2,4) ] ]
    all_funcs = exps + roots
    # all_funcs.append({'f': lambda x: x, 'dfdx': lambda x: 1})

    chunck_size = math.ceil(len(all_funcs) / cores)
    funcs_chuncks = b = [ all_funcs[math.ceil(chunck_size) * i : math.ceil(chunck_size) * (i+1)] for i in range(cores) ]
    results = []
    with Pool(cores) as p:
        results = p.map(latice_exploration_core_task, funcs_chuncks)
    # latice_exploration_core_task((funcs_chuncks[2]))



if __name__ == '__main__':
    print(latices_exploration())
