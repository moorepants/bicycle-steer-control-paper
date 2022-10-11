"""

This script creates plots of the eigenvalues and eigenvectors as a function of
speed for the linear Whipple-Carvallo model using realistic parameter values.
It also finds the speeds at which the system is uncontrollable with only a
steer input.

"""
import os

from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
import matplotlib.pyplot as plt
import numpy as np

from data import bike_with_rider, bike_without_rider
from model import SteerControlModel
from utils import find_uncontrollable_points

SCRIPT_PATH = os.path.realpath(__file__)
SRC_DIR = os.path.dirname(SCRIPT_PATH)
ROOT_DIR = os.path.realpath(os.path.join(SRC_DIR, '..'))
FIG_DIR = os.path.join(ROOT_DIR, 'figures')

if not os.path.exists(FIG_DIR):
    os.mkdir(FIG_DIR)

parameter_set = Meijaard2007ParameterSet(bike_with_rider, True)
model_with_rider = SteerControlModel(parameter_set)

fig, ax = plt.subplots()
ax = parameter_set.plot_all(ax=ax)
fig.savefig(os.path.join(FIG_DIR, 'uncontrolled-with-rider-geometry.png'),
            dpi=300)

fig, ax = plt.subplots()
model_with_rider.plot_eigenvalue_parts(ax=ax,
                                       v=np.linspace(0.0, 10.0, num=100))
points = find_uncontrollable_points(model_with_rider)
for point in points:
    ax.axvline(point, color='black')
ax.set_title('Uncontrollable speeds: {}'.format(points))
fig.savefig(os.path.join(FIG_DIR, 'uncontrolled-eigenvalues-with-rider.png'),
            dpi=300)

axes = model_with_rider.plot_eigenvectors(v=[0.0, 2.5, 5.0, 7.5, 10.0])
fig = axes[0, 0].figure
fig.savefig(os.path.join(FIG_DIR, 'uncontrolled-eigenvectors-with-rider.png'),
            dpi=300)

speeds = np.linspace(0.0, 10.0, num=1000)
axes = model_with_rider.plot_modal_controllability(acute=True, v=speeds)
for ax in axes[:, 1]:
    for point in points:
        ax.axvline(point, color='black')
fig = axes[0, 0].figure
fig.savefig(os.path.join(FIG_DIR, 'modal-controllability.png'), dpi=300)

parameter_set = Meijaard2007ParameterSet(bike_without_rider, False)
model_without_rider = SteerControlModel(parameter_set)

fig, ax = plt.subplots()
model_without_rider.plot_eigenvalue_parts(ax=ax, v=np.linspace(0.0, 10.0,
                                                               num=100))
points = find_uncontrollable_points(model_without_rider)
for point in points:
    ax.axvline(point, color='black')
ax.set_title('Uncontrollable speeds: {}'.format(points))
fig.savefig(os.path.join(FIG_DIR,
                         'uncontrolled-eigenvalues-without-rider.png'),
            dpi=300)

axes = model_without_rider.plot_eigenvectors(v=[0.0, 2.5, 5.0, 7.5, 10.0])
fig = axes[0, 0].figure
fig.savefig(os.path.join(FIG_DIR,
                         'uncontrolled-eigenvectors-without-rider.png'),
            dpi=300)

axes = model_with_rider.plot_mode_simulations(v=0.0)
fig = axes[0, 0].figure
fig.savefig(os.path.join(FIG_DIR, 'uncontrolled-mode-sims-v00.png'), dpi=300)

axes = model_with_rider.plot_mode_simulations(v=5.0)
fig = axes[0, 0].figure
fig.savefig(os.path.join(FIG_DIR, 'uncontrolled-mode-sims-v05.png'), dpi=300)

axes = model_with_rider.plot_mode_simulations(v=10.0)
fig = axes[0, 0].figure
fig.savefig(os.path.join(FIG_DIR, 'uncontrolled-mode-sims-v10.png'), dpi=300)
