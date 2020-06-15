import numpy as np
import matplotlib.pyplot as plt

# my ubuntu does not let me change matplotlib font, still need to check why.
labelsize = 14
fontsize  = 14

plt.tight_layout()

fig = plt.figure(1)
x = np.array([0.1962, 0.2943, 0.3924, 0.4905, 0.5886])
y = np.array([4.48, 5.983333333, 7.236666667, 7.923333333, 9.026666667])
# yerr = np.array(
#     [0.6324950593, 0.5841803375, 0.2515286597, 0.334090806, 0.1790716802])

# plt.errorbar(x, y, yerr=yerr, label='experimental displacement of the tip')

y_simu = np.array(
    [4.099640024, 6.149460036, 8.199280048, 10.24910006, 12.29892007])

plt.plot(x, y_simu, label='simulation displacement of the tip')


plt.rc('xtick', labelsize=labelsize)  # fontsize of the tick labels
plt.rc('ytick', labelsize=labelsize)
csfont = {'fontname':'Times New Roman'}
plt.xlabel('applied force (N)', fontsize=fontsize, **csfont)
plt.ylabel('tip displacement (mm)', fontsize=fontsize,**csfont)

plt.legend(loc='best', fontsize=fontsize)
# plt.show()
plt.savefig('plot_x_error_bar.pdf', dpi=200)
