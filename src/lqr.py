import os

from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
import control as ct
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

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

speeds = np.linspace(0.0, 10.0, num=1000)
gain_arrays = {'kphi': [], 'kphidot': [], 'kdelta': [], 'kdeltadot': []}
ctrb_mats = []
R = np.array([1.0])
Q = np.eye(4)
#Q = np.diag([1.0, 0.5, 10.0, 0.5])
for i, speed in enumerate(speeds):
    A, B = model.form_state_space_matrices(v=speed)
    B_delta = B[:, 1, np.newaxis]  # shape(4,1)
    C = ct.ctrb(A, B_delta)
    ctrb_mats.append(np.linalg.matrix_rank(C))
    #_, evals[i], K = ct.care(A, B_delta, Q)
    _, _, K = ct.care(A, B_delta, Q)
    K = K.squeeze()
    gain_arrays['kphi'].append(K[0])
    gain_arrays['kdelta'].append(K[1])
    gain_arrays['kphidot'].append(K[2])
    gain_arrays['kdeltadot'].append(K[3])
gain_arrays['kphi'] = np.array(gain_arrays['kphi'])
gain_arrays['kdelta'] = np.array(gain_arrays['kdelta'])
gain_arrays['kphidot'] = np.array(gain_arrays['kphidot'])
gain_arrays['kdeltadot'] = np.array(gain_arrays['kdeltadot'])

idxs = [0, 250, 500, 750, 999]
ax = model.plot_eigenvectors(v=speeds[idxs],
                             kphi=gain_arrays['kphi'][idxs],
                             kdelta=gain_arrays['kdelta'][idxs],
                             kphidot=gain_arrays['kphidot'][idxs],
                             kdeltadot=gain_arrays['kdeltadot'][idxs])

fig, ax = plt.subplots()
ax = model.plot_eigenvalue_parts(ax=ax, colors=4*['C0'], v=speeds)
model.plot_eigenvalue_parts(ax=ax, colors=4*['C1'], v=speeds, **gain_arrays)
fig.savefig(os.path.join(FIG_DIR, 'lqr-eig.png'), dpi=300)

fig, axes = plt.subplots(4, 1, sharex=True)
axes[0].plot(speeds, gain_arrays['kphi'])
axes[1].plot(speeds, gain_arrays['kdelta'])
axes[2].plot(speeds, gain_arrays['kphidot'])
axes[3].plot(speeds, gain_arrays['kdeltadot'])

speeds = np.linspace(0.0, 10.0, num=1000)
betas = np.rad2deg(model.calc_modal_controllability(v=speeds))
#betas[betas > 90.0] = 180.0 - betas[betas > 90.0]
fig, axes = plt.subplots(*betas[0].shape, sharex=True, sharey=True)
axes[0, 0].plot(speeds, betas[:, 0, 0])
axes[0, 1].plot(speeds, betas[:, 0, 1])
axes[1, 0].plot(speeds, betas[:, 1, 0])
axes[1, 1].plot(speeds, betas[:, 1, 1])
axes[2, 0].plot(speeds, betas[:, 2, 0])
axes[2, 1].plot(speeds, betas[:, 2, 1])
axes[3, 0].plot(speeds, betas[:, 3, 0])
axes[3, 1].plot(speeds, betas[:, 3, 1])
#fig, ax = plt.subplots()
#ax.plot(speeds, betas[:, 0, 0], '.')
#ax.plot(speeds, betas[:, 0, 1], '.')
#ax.plot(speeds, betas[:, 1, 0], '.')
#ax.plot(speeds, betas[:, 1, 1], '.')
#ax.plot(speeds, betas[:, 2, 0], '.')
#ax.plot(speeds, betas[:, 2, 1], '.')
#ax.plot(speeds, betas[:, 3, 0], '.')
#ax.plot(speeds, betas[:, 3, 1], '.')
#ax.set_ylim((0.0, 90.0))
fig.savefig(os.path.join(FIG_DIR, 'modal-controllability.png'), dpi=300)

plt.show()
