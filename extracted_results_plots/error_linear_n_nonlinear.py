import numpy as np
import matplotlib.pyplot as plt
#####################
## Error analysis##
#####################
# my ubuntu does not let me change matplotlib font, still need to check why.
labelsize = 12
fontsize  = 13

plt.tight_layout()

fig = plt.figure(1)
x = np.array([1, 2, 3])
y = np.array([0.000284093770721, 7.70404750370472e-06, 3.07971221710008E-08])
plt.semilogy(x, y )
plt.xlim([1,3])
plt.xticks([1,2,3])
plt.ylim([1e-8, 5e-4])

x = np.array([1, 2, 3])
y = np.array([1.47250305256942E-05, 2.46934688863197E-05, 5.15681102311087E-06])
plt.semilogy(x, y )
plt.xlim([1,3])
plt.xticks([1,2,3])
plt.ylim([1e-8, 5e-4])

plt.rc('xtick', labelsize=labelsize)  # fontsize of the tick labels
plt.rc('ytick', labelsize=labelsize)
csfont = {'fontname':'Times New Roman'}
plt.xlabel('POD basis', fontsize=fontsize, **csfont)
plt.ylabel('Reduction error (geometric mean)', fontsize=fontsize,**csfont)
plt.gca().tick_params(axis = 'both', which = 'major', labelsize = labelsize)
plt.gca().tick_params(axis = 'both', which = 'minor', labelsize = labelsize)

plt.legend(['POD','POD-DEIM'],loc='best', fontsize=fontsize)
# plt.show()
plt.savefig('error.pdf', dpi=200)

#####################
## Speedup analysis##
#####################
plt.tight_layout()

fig = plt.figure(2)
x = np.array([1, 2, 3])
y = np.array([2.46932148447877,2.25166787929635, 2.23675207938729])
plt.semilogy(x, y )
plt.xlim([1,3])
plt.xticks([1,2,3])
plt.ylim([1e0, 1e2])

x = np.array([1, 2, 3])
y = np.array([54.7276771188214, 8.24941386026471, 61.6007145916647])
plt.semilogy(x, y )
plt.xlim([1,3])
plt.xticks([1,2,3])
plt.ylim([1e0, 1e2])

plt.rc('xtick', labelsize=labelsize)  # fontsize of the tick labels
plt.rc('ytick', labelsize=labelsize)
csfont = {'fontname':'Times New Roman'}
plt.xlabel('POD basis', fontsize=fontsize, **csfont)
plt.ylabel('Speed increase (times) compare to the FOM', fontsize=fontsize,**csfont)
plt.gca().tick_params(axis = 'both', which = 'major', labelsize = labelsize)
plt.gca().tick_params(axis = 'both', which = 'minor', labelsize = labelsize)

plt.legend(['POD','POD-DEIM'],loc='best', fontsize=fontsize)
# plt.show()
plt.savefig('speed.pdf', dpi=200)