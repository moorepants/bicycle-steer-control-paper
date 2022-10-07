"""

This script creates plots of the eigenvalues and eigenvectors as a function of
speed for the linear Whipple-Carvallo model using realistic parameter values.
It also finds the speeds at which the system is uncontrollable with only a
steer input.

"""
import os

from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
import control as ct
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

from data import bike_with_rider, bike_without_rider
from model import SteerControlModel

SCRIPT_PATH = os.path.realpath(__file__)
SRC_DIR = os.path.dirname(SCRIPT_PATH)
ROOT_DIR = os.path.realpath(os.path.join(SRC_DIR, '..'))
FIG_DIR = os.path.join(ROOT_DIR, 'figures')

if not os.path.exists(FIG_DIR):
    os.mkdir(FIG_DIR)

parameter_set = Meijaard2007ParameterSet(bike_with_rider, True)
model_with_rider = SteerControlModel(parameter_set)


def calc_det_control(model, speed):
    A, B = model.form_state_space_matrices(v=speed)
    # TODO : This squeeze shouldn't be necessary.
    A, B, = np.squeeze(A), np.squeeze(B)
    # only steer control
    B_delta = B[:, 1, np.newaxis]  # shape(4,1)
    C = ct.ctrb(A, B_delta)
    return np.linalg.det(C)


def find_uncontrollable_points(model):
    uncontrollable_points = []
    for v in np.linspace(0.0, 10.0, num=20):
        res = spo.fsolve(lambda sp: calc_det_control(model, sp), v, xtol=1e-12)
        uncontrollable_points.append(res[0])
    return np.unique(np.round(uncontrollable_points, 5))


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
