import numpy as np 
from scipy.optimize import OptimizeResult
from .utils import _status_message, vecnorm, wrap_function



def minimize_adam(fun, x1, args=(), jac=None, beta1=0.9, beta2=0.99, eps=1e-8, lr=0.01,
                   callback=None, gtol=1e-5, norm=np.Inf, maxiter=None, disp=False, **unknown_options):
    f = fun
    fprime = jac

    x1 = np.asarray(x1).flatten()
    if x1.ndim == 0:
        x1.shape = (1,)
    if maxiter is None:
        maxiter = len(x1) * 500
    func_calls, f = wrap_function(f, args)
    grad_calls, myfprime = wrap_function(fprime, args)

    # initialization
    k = 1
    gfk = myfprime(x1)
    mk = (1-beta1) * gfk
    vk = (1-beta2) * (gfk**2)
    gnormk = vecnorm(gfk, ord=norm)
    warnflag = 0
    
    # iteration step
    old_fval = f(x1)
    xk = x1
    while (gnormk > gtol) and (k <= maxiter):
        if not np.isfinite(old_fval):
            warnflag = 2
            break
        mk_hat = mk / (1 - beta1**k)
        vk_hat = vk / (1 - beta2**k)
        xk = xk - lr * mk_hat / (np.sqrt(vk_hat) + eps)
        gfk = myfprime(xk)
        gnormk = vecnorm(gfk)
        if (gnormk <= gtol):
            break
        mk = beta1 * mk + (1-beta1) * gfk
        vk = beta2 * vk + (1-beta2) * (gfk**2)
        k += 1
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