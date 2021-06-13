import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

stability = np.loadtxt("stability.csv",delimiter=',',dtype=int)
largest_eigen = np.loadtxt("largest_multiplier_abs.csv",delimiter=',',dtype=float)

print(stability)
print(largest_eigen.shape)

params = open("extraparams.txt",'r')
plot_title = next(params)
plot_xlabel = next(params)
plot_ylabel = next(params)
plot_xs = [float(x) for x in next(params).split()]
plot_xe = [float(x) for x in next(params).split()]
nstep = int(next(params))

param = np.linspace(plot_xs,plot_xe,nstep)
plt.plot(param,largest_eigen)
plt.title(plot_title)
plt.xlabel(plot_xlabel)
plt.ylabel(plot_ylabel)
plt.savefig("Largest_Eigen.png",dpi=300)
plt.show()
