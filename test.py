import numpy as np
from scipy.optimize import minimize
# from adam import minimize_adam
from gd import minimize_gd
from lr_scheduler import StepScheduler
import matplotlib.pyplot as plt 

from benchmark import beale, beale_der

# benchmark using Rosenbrock funciton
## rosen
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
XY = np.stack([X.ravel(), Y.ravel()], axis=1)
Z = (np.array([beale(xy) for xy in XY])).reshape([100, 100])

## optimize
x0 = np.array([1.49355, 2.66234])

info = [x0]

def callback(x):
    global info
    info.append(x)

step_lr = StepScheduler(0.5, 10)
res = minimize(beale, x0, method=minimize_gd, jac=beale_der,
               callback=callback, options={'maxiter': 2000,
                                           'lr': 0.0001,
                                           'nesterov': True,
                                           'lr_scheduler': step_lr})
print(res.success)
x = res.x 
print(x)

## plot
xs = np.array(info)
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.contour(X, Y, Z, np.arange(10)**5)
plt.plot(xs[:, 0], xs[:, 1], '-o')
plt.subplot(122)

beale_vals = np.array([beale(x) for x in xs])
plt.semilogy(range(len(xs)), beale_vals)
plt.show()