import numpy as np
from scipy.differentiate import derivative
import matplotlib.pyplot as plt

def diff_central(f, x0, h):
	return (f(x0+h) - f(x0-h))/(2*h)

def diff_scipy(f, x0, h):
	res = derivative(f, x0, maxiter=1, order=2, initial_step=h)
	return res.df

def diff_scipy_def(f, x0, h):
	res = derivative(f, x0)
	return res.df

def errores(f, x0, df_exact, df_estim, h_values):
	errores = []
	for h in h_values:
		errores.append(np.abs(df_exact(x0) - df_estim(f, x0, h)))
	return errores

j_values = np.arange(20)+1

h_values = [10.**-j for j in j_values]

f = lambda x : np.sin(x)

df = lambda x : np.cos(x)

#x0 = 0.4
x0 = np.pi/2

err_cent = errores(f, x0, df, diff_central, h_values)
err_scipy = errores(f, x0, df, diff_scipy, h_values)
err_scipy_def = errores(f, x0, df, diff_scipy_def, h_values)

print(df(x0))
print(diff_scipy_def(f, x0, 1))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(j_values, err_cent, label="central")
ax.plot(j_values, err_scipy, "*", label="scipy")
ax.plot(j_values, err_scipy_def, label="scipy_def")

ax.set_yscale('log')

plt.legend()
plt.grid()
plt.show()
