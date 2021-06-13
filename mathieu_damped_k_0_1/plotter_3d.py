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
plot_ye, plot_xs = [float(x) for x in next(params).split()]
plot_ys, plot_xe = [float(x) for x in next(params).split()]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig,ax = plt.subplots(1,1)
cp = ax.imshow(stability, extent = [plot_xs, plot_xe, plot_ys, plot_ye], cmap='gray')
fig.colorbar(cp)
plt.title(plot_title)
plt.xlabel(plot_xlabel)
plt.ylabel(plot_ylabel)
ax.set_aspect('auto')
plt.savefig("Stability.png",dpi=300)
plt.show()


fig,ax = plt.subplots(1,1)
cp = ax.imshow(largest_eigen-1., extent = [plot_xs, plot_xe, plot_ys, plot_ye], cmap='gray', norm=colors.SymLogNorm(1e-3,base=10)) #vmin = min(np.min(largest_eigen),0.00001),
fig.colorbar(cp)
plt.title(plot_title + r" (Plotting $\rho_{max}-1$)")
plt.xlabel(plot_xlabel)
plt.ylabel(plot_ylabel)
ax.set_aspect('auto')
plt.savefig("Largest_Eigen.png",dpi=300)
plt.show()
