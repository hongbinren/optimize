import numpy as np 


_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.'}


def vecnorm(x, ord=2):
    if ord == np.Inf:
        return np.amax(np.abs(x))
    elif ord == -np.Inf:
        return np.amin(np.abs(x))
    else:
        return np.sum(np.abs(x)**ord, axis=0)**(1.0 / ord)

def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper