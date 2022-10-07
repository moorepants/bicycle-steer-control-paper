import os

from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from data import bike_with_rider
from model import SteerControlModel

SCRIPT_PATH = os.path.realpath(__file__)
SRC_DIR = os.path.dirname(SCRIPT_PATH)
ROOT_DIR = os.path.realpath(os.path.join(SRC_DIR, '..'))
FIG_DIR = os.path.join(ROOT_DIR, 'figures')

if not os.path.exists(FIG_DIR):
    os.mkdir(FIG_DIR)

parameter_set = Meijaard2007ParameterSet(bike_with_rider, True)
model = SteerControlModel(parameter_set)

# Figure 1: Plot the real & imaginary eigenvalue parts as a function of speed
# for a series of roll derivative control gains.
cmap = mpl.colormaps['viridis']
gains = [0.0, -5.0, -10.0, -50.0, -100.0, -500.0]  # , -1000.0, -5000.0]
color_vals = np.linspace(0.2, 1.0, num=len(gains))

fig, ax = plt.subplots()
for gain, color_val in zip(gains, color_vals):
    color_rgb = cmap(color_val)
    ax = model.plot_eigenvalue_parts(ax=ax, kphidot=gain,
                                     colors=4*[color_rgb],
                                     v=np.linspace(0.0, 10.0, num=400))
ax.set_ylim((-10.0, 10.0))

fig.savefig(os.path.join(FIG_DIR, 'roll-rate-eig-effect.png'), dpi=300)

# Figure 2: At a specific low speed how does ramping up the roll rate gain
# change the eigenvalues.
fig, ax = plt.subplots()
ax = model.plot_eigenvalue_parts(ax=ax,
                                 kphidot=np.linspace(0.0, -50.0, num=100),
                                 v=1.0)
fig.savefig(os.path.join(FIG_DIR, 'roll-rate-low-speed-eig-effect.png'),
            dpi=300)

# Figure 3: Show what the eigenvectors of the closed loop control look like at
# different speeds.
axes = model.plot_eigenvectors(kphidot=-50.0,
                               v=[0.0, 2.5, 5.0, 7.5, 10.0])
fig = axes[0, 0].figure
fig.savefig(os.path.join(FIG_DIR, 'roll-rate-gain-evec-effect.png'),
            dpi=300)
