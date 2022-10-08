import os

from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
import matplotlib.pyplot as plt
import numpy as np
import control as ct

from data import bike_with_rider, bike_without_rider
from model import SteerControlModel

SCRIPT_PATH = os.path.realpath(__file__)
SRC_DIR = os.path.dirname(SCRIPT_PATH)
ROOT_DIR = os.path.realpath(os.path.join(SRC_DIR, '..'))
FIG_DIR = os.path.join(ROOT_DIR, 'figures')

if not os.path.exists(FIG_DIR):
    os.mkdir(FIG_DIR)

parameter_set = Meijaard2007ParameterSet(bike_with_rider, True)
model = SteerControlModel(parameter_set)

evals, evecs = model.calc_eigen(v=6.0)

speeds = np.linspace(0.0, 10.0, num=1000)

gains = np.empty((len(speeds), 1, 4))

for i, v in enumerate(speeds):
    A, B = model.form_state_space_matrices(v=v)
    K = ct.acker(A, B[:, 1, np.newaxis], evals)
    gains[i] = K

fig, ax = plt.subplots()
model.plot_eigenvalue_parts(ax=ax,
                            v=speeds,
                            kphi=gains[:, 0, 0],
                            kdelta=gains[:, 0, 1],
                            kphidot=gains[:, 0, 2],
                            kdeltadot=gains[:, 0, 3])
fig.savefig(os.path.join(FIG_DIR, 'pole-place-v06.png'), dpi=300)

fig, axes = plt.subplots(4)
axes[0].plot(speeds, gains[:, 0, 0])
axes[1].plot(speeds, gains[:, 0, 1])
axes[2].plot(speeds, gains[:, 0, 2])
axes[3].plot(speeds, gains[:, 0, 3])
fig.savefig(os.path.join(FIG_DIR, 'pole-place-v06-gains.png'), dpi=300)

param_without_rider = Meijaard2007ParameterSet(bike_without_rider, False)
model_without_rider = SteerControlModel(param_without_rider)

gains = np.empty((len(speeds), 1, 4))

for i, v in enumerate(speeds):
    evals, evecs = model_without_rider.calc_eigen(v=v)
    A, B = model.form_state_space_matrices(v=v)
    K = ct.acker(A, B[:, 1, np.newaxis], evals)
    gains[i] = K

fig, ax = plt.subplots()
ax = model_without_rider.plot_eigenvalue_parts(ax=ax, colors=4*['grey'],
                                               v=speeds)
model.plot_eigenvalue_parts(ax=ax,
                            v=speeds,
                            kphi=gains[:, 0, 0],
                            kdelta=gains[:, 0, 1],
                            kphidot=gains[:, 0, 2],
                            kdeltadot=gains[:, 0, 3])
fig.savefig(os.path.join(FIG_DIR, 'pole-place-match-other-bike-evals.png'),
            dpi=300)

fig, axes = plt.subplots(4)
axes[0].plot(speeds, gains[:, 0, 0])
axes[1].plot(speeds, gains[:, 0, 1])
axes[2].plot(speeds, gains[:, 0, 2])
axes[3].plot(speeds, gains[:, 0, 3])
fig.savefig(os.path.join(FIG_DIR, 'pole-place-match-other-bike-gains.png'),
            dpi=300)
