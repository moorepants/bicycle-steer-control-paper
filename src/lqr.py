import os

from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
import control as ct
import matplotlib.pyplot as plt
import numpy as np

from data import bike_with_rider
from model import SteerControlModel
from utils import find_uncontrollable_points

SCRIPT_PATH = os.path.realpath(__file__)
SRC_DIR = os.path.dirname(SCRIPT_PATH)
ROOT_DIR = os.path.realpath(os.path.join(SRC_DIR, '..'))
FIG_DIR = os.path.join(ROOT_DIR, 'figures')

if not os.path.exists(FIG_DIR):
    os.mkdir(FIG_DIR)

parameter_set = Meijaard2007ParameterSet(bike_with_rider, True)
model = SteerControlModel(parameter_set)

points = find_uncontrollable_points(model)

max_torque = 10.0
max_states = 0.20*np.deg2rad([20.0, 100.0, 45.0, 300.0])

speeds = np.linspace(0.0, 10.0, num=1000)
gains = {'kphi': np.empty_like(speeds),
         'kphidot': np.empty_like(speeds),
         'kdelta': np.empty_like(speeds),
         'kdeltadot': np.empty_like(speeds)}
ctrb_mats = []
Q = np.eye(4)
# Q = np.diag([1.0, 0.5, 10.0, 0.5])
for i, speed in enumerate(speeds):
    A, B = model.form_state_space_matrices(v=speed)
    B_delta = B[:, 1, np.newaxis]  # shape(4,1)
    ctrb_mats.append(ct.ctrb(A, B_delta))
    _, _, K = ct.care(A, B_delta, Q)
    # gains can't produce too much torque!
    while np.abs(K@max_states) > max_torque:
        K = 0.95*K
    K = K.squeeze()
    gains['kphi'][i] = K[0]
    gains['kdelta'][i] = K[1]
    gains['kphidot'][i] = K[2]
    gains['kdeltadot'][i] = K[3]

fig, ax = plt.subplots()
ax = model.plot_eigenvalue_parts(ax=ax, colors=4*['C0'], v=speeds)
model.plot_eigenvalue_parts(ax=ax, colors=4*['C1'], v=speeds, **gains)
fig.savefig(os.path.join(FIG_DIR, 'lqr-eig.png'), dpi=300)

idxs = [0, 250, 500, 750, 999]
axes = model.plot_eigenvectors(v=speeds[idxs],
                               kphi=gains['kphi'][idxs],
                               kdelta=gains['kdelta'][idxs],
                               kphidot=gains['kphidot'][idxs],
                               kdeltadot=gains['kdeltadot'][idxs])

fig = axes[0, 0].figure
fig.savefig(os.path.join(FIG_DIR, 'lqr-evec.png'), dpi=300)

axes = model.plot_mode_simulations(v=speeds[idxs[0]],
                                   kphi=gains['kphi'][idxs[0]],
                                   kdelta=gains['kdelta'][idxs[0]],
                                   kphidot=gains['kphidot'][idxs[0]],
                                   kdeltadot=gains['kdeltadot'][idxs[0]])
fig = axes[0, 0].figure
fig.savefig(os.path.join(FIG_DIR, 'lqr-mode-sims-v00.png'), dpi=300)

axes = model.plot_mode_simulations(v=speeds[idxs[2]],
                                   kphi=gains['kphi'][idxs[2]],
                                   kdelta=gains['kdelta'][idxs[2]],
                                   kphidot=gains['kphidot'][idxs[2]],
                                   kdeltadot=gains['kdeltadot'][idxs[2]])
fig = axes[0, 0].figure
fig.savefig(os.path.join(FIG_DIR, 'lqr-mode-sims-v05.png'), dpi=300)

axes = model.plot_mode_simulations(v=speeds[idxs[4]],
                                   kphi=gains['kphi'][idxs[4]],
                                   kdelta=gains['kdelta'][idxs[4]],
                                   kphidot=gains['kphidot'][idxs[4]],
                                   kdeltadot=gains['kdeltadot'][idxs[4]])
fig = axes[0, 0].figure
fig.savefig(os.path.join(FIG_DIR, 'lqr-mode-sims-v10.png'), dpi=300)

fig, axes = plt.subplots(4, 1, sharex=True)
for i, (k, v) in enumerate(gains.items()):
    for point in points:
        axes[i].axvline(point, color='black')
    axes[i].plot(speeds, v)
    axes[i].set_ylabel(k)
    axes[i].grid()
    #axes[i].set_ylim((-1000.0, 1000.0))
fig.savefig(os.path.join(FIG_DIR, 'lqr-gains.png'), dpi=300)

idx = 100
times = np.linspace(0.0, 10.0, num=1000)
def controller(t, x, par):
    K = np.array([[0.0, 0.0, 0.0, 0.0],
                  [gains['kphi'][idx], gains['kdelta'][idx],
                   gains['kphidot'][idx], gains['kdeltadot'][idx]]])
    torques = -K@x
    #if np.abs(torques[1]) >= 10.0:
        #torques[1] = np.sign(torques[1])*10.0
    return torques
x0 = np.deg2rad([10.0, -10.0, 0.0, 0.0])
axes = model.plot_simulation(times, x0, input_func=controller, v=speeds[idx])
axes[0].set_title('Speed = {:1.2f}'.format(speeds[idx]))
fig = axes[0].figure
fig.savefig(os.path.join(FIG_DIR, 'lqr-simulation.png'),
            dpi=300)
