import datetime
from random import sample
import os
import pickle
import time
import copy
import PIL
from random import sample as random_sample

import cma
import numpy as np
import telegram
import yaml
from joblib import Parallel, delayed


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    while True:
        indices = sorted(sample(range(n), r))
        yield tuple(pool[i] for i in indices)

def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate

def cmaes_optimize(estimator_cls, X, y, load_imgs,
    param_space, fixed_variables, initial_params,
    **kwargs):
    """ Optimizes the parameters of estimator using the CMA-ES algorithm.

    Arguments:
        estimator_cls (class): Estimator class on which parameters optimization should be performed
        reloadDS (function): Function that returns or yields (preferable for large datasets) pairs of (X, y), where X are the inputs for the estimator and y are its ground truth values
        param_space (dict): Maps parameter names, even those fixed, to their bounds (es. {name: [min, max], ...})
        fixed_variables (dict): Maps parameter names to their fixed values
        initial_params (dict): Maps parameter names to their initial values (must be within the bounds)
        sample_size (int): How many samples should be picked from the dataset at each step, or negative values to take the whole of it (defaults to 10)
        n_iter (int): How many iterations should be performed (defaults to 10)
        n_jobs (int): How many processes should be used for parallel evaluations, or a negative value to use as many threads as possible (defaults to -1)
        verbose (bool): Whether information about optimization should be sent (defaults to False)
        verboseTelegram (bool): Whether information about optimization should be sent (defaults to True)
        load (str): Path to a saved instance of the CMA-ES optimizer that needs to be loaded (defaults to None - i.e. no effect)
        outDir (str): Path to the output folder, where the partial and final results will be saved (defaults to the current directory)
        retFuns (bool): When True, besides the optimization current state, returns also the functions

            - encode, encodes the values of all variables by normalization within their bounds; takes as its only argument a dictionary that maps parameter names to their values;
            - decode, inverse function of encode, and takes the same arguments;
            - fit_fun, evaluates a set of parameters, and takes as its only argument a list of parameter values whose order corresponds to nonfixed_keys;
            - disp_cma_results, returns a description of the current CMA state, and takes as arguments the current state of the optimizer (as returned by this function), the decode function, and the list of parameter names (see nonfixed_keys);
            - nonfixed_keys, the list of names of independent parameters, i.e. the optimized ones.

    Returns:
        The current state of the optimizer, if retFuns is False, otherwise check the retFuns argument.
    """
    sample_size = kwargs.get('sample_size', 10)
    n_iter = kwargs.get('n_iter', 10)
    n_jobs = kwargs.get('n_jobs', -1)
    verbose = kwargs.get('verbose', False)
    verboseTelegram = kwargs.get('verboseTelegram', True)
    load = kwargs.get('load', None)
    outDir = kwargs.get('outDir', os.path.abspath(os.path.curdir))
    retFuns = kwargs.get('retFuns', True)

    # Get a list of independent keys
    nonfixed_keys = [key for key in param_space if key not in fixed_variables]

    # Automatically determine the type of the parameter, based on the input type
    nonfixed_types = list(map(type, (v for k,v in initial_params.items() if k in nonfixed_keys)))
    type_fixing = lambda kwa: {kv[0]:t(kv[1]) for kv, t in zip(kwa.items(), nonfixed_types)}

    @static_var('prev_time', 0)
    def disp_cma_results(es, scale=None, names=None):
        """ Returns a description of the current CMA state """
        if scale is None:
            scale = lambda x: x
        fmt_res = "\
*Iter:* {}, with {} evaluations\n\
*Total time elapsed:* {}\n\
*Previous iteration time elapsed:* {}\n\
*Current best solution:* \n\t- {}\n\
*Current function values:* {}\
        ".format(
            es.countiter, es.countevals, 
            datetime.timedelta(seconds=int(es.timer.elapsed)),
            datetime.timedelta(seconds=int(es.timer.elapsed-disp_cma_results.prev_time)),
            '\n\t- '.join( '_{}_ = {}'.format(key.replace('_',' '), val) 
                for key, val in type_fixing(scale(dict(zip(names, es.pop_sorted[0])))).items() ),
            np.array2string(es.fit.fit, precision=4, separator=', ')
        )
        disp_cma_results.prev_time = es.timer.elapsed
        return fmt_res

    def encode(kwa: dict):
        """ Encodes the values of all variables by normalization within their bounds.
        
        Args:
            kwa: {name: value} dictionary of parameters
        """
        kwac = kwa.copy()
        for key in kwac:
            kwac[key] -= param_space[key][0]
            kwac[key] /= param_space[key][1]-param_space[key][0]
        return kwac

    def decode(kwa: dict):
        """ Decodes the values of all variables by undoing normalization.
        
        Args:
            kwa: {name: value} dictionary of parameters
        """
        kwac = kwa.copy()
        for key in kwac:
            kwac[key] *= param_space[key][1]-param_space[key][0]
            kwac[key] += param_space[key][0]
        return kwac

    # Define the logic for the creation of the sub-datasets
    if sample_size < 0:
        XX, yy = zip(*load_imgs(X, y))
        XX = list(XX)
        yy = list(yy)
    else:
        Xy = random_combination( zip(X, y), sample_size ) # no replacement

    def fit_fun(x):
        """ Evaluates the set of parameters x.
        
        Args:
            x: list of values whose order corresponds to nonfixed_keys
        """
        kwa = dict(zip(nonfixed_keys, x))
        kwa = decode(kwa)
        # Convert to appropriate types
        kwa = type_fixing(kwa)
        # Add the fixed variables before the computation
        kwa.update(fixed_variables)
        return 1.0 - estimator_cls(**kwa).score( XX, yy )
    
    if load is None:
        # Get the initial set of parameters, according to the current parameters space and fixed variables
        x0 = {k:v for k,v in initial_params.items() if k in nonfixed_keys}
        # The values of each solution will be sorted as the keys of nonfixed_keys
        x0 = [x0[k] for k in nonfixed_keys]
        # Normalize the initial values
        x0 = list(encode(dict(zip(nonfixed_keys, x0))).values())

        options = {
            'bounds': [0, 1],
            'BoundaryHandler': cma.BoundTransform,
            'tolfun': 1e-4,
            'tolx': 1e-5,
            'verb_log': -1,
            'maxiter': n_iter,
        }
        search_results = cma.CMAEvolutionStrategy(x0, 0.25, inopts=options)
    else:
        with open(load, 'rb') as f:
            search_results = pickle.load(f)

    if verboseTelegram:
        try:
            initial_message = "<b>Started parameters optimization:</b>\n<pre>\n\n"
            trunc = 30
            w1 = max([len(key) for key in kwargs.keys()])
            w2 = max([len(str(val)[-trunc:]) for val in kwargs.values()])
            rowfmt = "| {:>{w1}} | {:{w2}} |\n"
            initial_message += rowfmt.format("Key", "Value", w1=w1, w2=w2)
            initial_message += "| {:{c}^{w1}} | {:{c}^{w2}} |\n".format("", "", w1=w1, w2=w2, c='-')
            for key, val in kwargs.items():
                initial_message += rowfmt.format(
                    key.replace('_', ' '),
                    str(val)[-trunc:].replace('_', ' '),
                    w1=w1, w2=w2)
            initial_message += "</pre>"
            bot = telegram.Bot(token='546794449:AAGzmfH9Oa6277Vsl2T9hRrGnNHHSpEMsd8')
            bot.send_message(chat_id=41795159, text=initial_message, parse_mode=telegram.ParseMode.HTML)
        except Exception as err:
            print("Cannot send message to Bot due to", err)

    # Get the number of process that should spawn
    n_processes = search_results.popsize + 1 if n_jobs < 0 else n_jobs
    
    try:
        with Parallel(n_processes) as parallel:
            while not search_results.stop():
                # Load batch
                if sample_size > 0:
                    try:
                        X, y = zip(*next(Xy))
                    except IndexError:
                        break
                    XX, yy = zip(*load_imgs(X, y))
                    XX = list(XX)
                    yy = list(yy)
                # Sample some solutions
                solutions = search_results.ask()
                # --- verbose ---
                if (np.array(solutions) < 0).any() or (np.array(solutions) > 1).any():
                    log = "Solutions out of bounds..."
                    if verbose:
                        print(log)
                    if verboseTelegram:
                        try:
                            bot.send_message(chat_id=41795159, text=log, parse_mode=telegram.ParseMode.MARKDOWN)
                        except Exception as err:
                            print("Cannot send message to Bot due to", err)
                            continue
                # ---------------
                # Test them and compute the update
                evals = parallel(delayed(fit_fun)(_x) for _x in solutions)
                search_results.tell(solutions, evals)
                # Save the parameter search results
                curiter = search_results.result.iterations
                curfile = "_{}".format(curiter).join(os.path.splitext(outDir))
                with open(curfile, "wb") as f:
                    pickle.dump(search_results, f)
                curpar = os.path.splitext(outDir)[0] + "_bestpar_{}.yml".format(curiter)
                with open(curpar, 'w') as f:
                    yaml.dump(type_fixing(decode(dict(zip(nonfixed_keys, search_results.result.xbest.tolist())))), f, Dumper=yaml.Dumper)
                # --- verbose ---
                print('{{"metric": "Best Score", "value": {}, "step": {}}}'.format(search_results.best.f, curiter))
                log = disp_cma_results(search_results, scale=decode, names=nonfixed_keys)
                if verbose:
                    print(log)
                if verboseTelegram:
                    try:
                        bot.send_message(chat_id=41795159, text=log, parse_mode=telegram.ParseMode.MARKDOWN)
                    except Exception as err:
                        print("Cannot send message to Bot due to", err)
                        continue
                # ---------------
    except KeyboardInterrupt:
        print("Interrupted by the user")

    if retFuns:
        return search_results, encode, decode, fit_fun, disp_cma_results, nonfixed_keys
    else:
        return search_results
