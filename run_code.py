import itertools
import find_params_basic_alg
import math
import multiprocessing

def latices_exploration(cores=8):
    exps = [ {'f': lambda x: n**x, 'dfdx': lambda x: n.ln() * n**x} for n in [ dec(i) for i in range(2,22) ] ]
    roots = [ {'f': lambda x: x**q, 'dfdx': lambda x: q * x**(q-1)} for q in [ dec(1)/dec(i) for i in range(2,10) ] ]
    all_funcs = exps + roots
    all_funcs.append({'f': lambda x: x, 'dfdx': lambda x: 1})

    chunck_size = math.ceil(len(all_funcs) / cores)
    funcs_chuncks = b = [ all_funcs[math.ceil(chunck_size * i : chunck_size * (i+1)] for i in range(cores) ]
    results = []
    with Pool(cores) as p:
        results = p.map(latice_exploration_core_task, funcs_chuncks)

def latice_exploration_core_task(funcs_chunck):
    results = []
    for f_dict in funcs_chunck:
        gd = find_params_basic_alg.GradientDescentBasicAlgo(enforce_Z=True, **f_dict)
        for x,y in itertools.product(range(1,50, range(1,50))):
            res = gd.find_params(x, y, show_progress=False)
            if res:
                results.append({'values': res, 'f_dict': f_dict})
    return results

