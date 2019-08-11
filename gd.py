import numpy as np
from inspect import isgenerator
from scipy.optimize import OptimizeResult 
from utils import _status_message, vecnorm, wrap_function


def minimize_gd(fun, x1, args=(), jac=None, momentum=0.95, lr=0.0001,
                nesterov=False, lr_scheduler=None,
                callback=None, gtol=1e-5, norm=np.Inf, maxiter=None,
                disp=False, **unknown_options):
    f = fun
    fprime = jac
    if (lr_scheduler is not None) and callable(lr_scheduler):
        lr_ = lr_scheduler(lr)
    else:
        lr_ = lr

    x1 = np.asarray(x1).flatten()
    if x1.ndim == 0:
        x1.shape = (1,)
    if maxiter is None:
        maxiter = len(x1) * 500
    func_calls, f = wrap_function(f, args)
    grad_calls, myfprime = wrap_function(fprime, args)

    k = 1
    gfk = myfprime(x1)
    if isgenerator(lr_):
        vk = next(lr_) * gfk
    else:
        vk = lr_ * gfk
    gnormk = vecnorm(gfk, ord=norm)
    warnflag = 0

    old_fval = f(x1)
    xk = x1
    while (gnormk > gtol) and (k <= maxiter):
        if not np.isfinite(old_fval):
            warnflag = 2
            break
        xk = xk - vk
        gfk = myfprime(xk)
        gnormk = vecnorm(gfk)
        if (gnormk <= gtol):
            break
        k += 1
        if nesterov:
            _gfk_ahead = myfprime(xk - momentum * vk)
            if isgenerator(lr_):
                vk = momentum * vk + next(lr_) * _gfk_ahead
            else:
                vk = momentum * vk + lr_ * _gfk_ahead
        else:
            if isgenerator(lr_):
                vk = momentum * vk + next(lr_) * gfk
            else:
                vk = momentum * vk + lr_ * gfk
        old_fval = f(xk)
        if callback is not None:
            callback(xk)

    fval = old_fval
    
    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % func_calls[0])
        print("         Gradient evaluations: %d" % grad_calls[0])


    result = OptimizeResult(fun=fval, jac=gfk, nfev=func_calls[0],
                            njev=grad_calls[0], status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)

    return result