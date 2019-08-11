import numpy as np 
from scipy.optimize import rosen, rosen_der


__all__ = ['rosen', 'rosen_der', 'beale', 'beale_der']


def beale(x):
    x1 = x[0]
    x2 = x[1]
    return (1.5 - x1 + x1 * x2)**2 +\
           (2.25 - x1 + x1 * (x2**2))**2 +\
           (2.625 - x1 + x1 * (x2**3))**2

def beale_der(x):
    x1 = x[0]
    x2 = x[1]
    der1 = 2 * ((1.5 - x1 + x1 * x2) * (x2 - 1) +\
           (2.25 - x1 + x1 * (x2**2)) * (x2**2 - 1) +\
           (2.625 - x1 + x1 * (x2**3)) * (x2**3 - 1))
    der2 = 2 * ((1.5 - x1 + x1 * x2) * x1 +\
           (2.25 - x1 + x1 * (x2**2)) * (2*x1*x2) +\
           (2.625 - x1 + x1 * (x2**3)) * (3*x1*(x2**2)))
    return np.array([der1, der2])


if __name__ == "__main__":
    
    from scipy.optimize import check_grad

    X = Y = np.random.uniform(-2, 2, 10)
    XX, YY = np.meshgrid(X, Y)
    XY = np.stack([XX.ravel(), YY.ravel()], axis=1)
    Errs = []
    for xy in XY:
        grad_err = check_grad(beale, beale_der, xy)
        err = np.less(grad_err, 1e-4)
        Errs.append(err)
    assert all(Errs) is True, print('derivative may calculate wrong')
    

