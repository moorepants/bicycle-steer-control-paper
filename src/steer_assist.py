import os

from scipy.linalg import solve_continuous_are
import control as ct
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from bicycleparameters.bicycle import benchmark_par_to_canonical
from bicycleparameters.models import Meijaard2007Model
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet

SCRIPT_PATH = os.path.realpath(__file__)
SRC_DIR = os.path.dirname(SCRIPT_PATH)
ROOT_DIR = os.path.realpath(os.path.join(SRC_DIR, '..'))
FIG_DIR = os.path.join(ROOT_DIR, 'figures')


class SteerAssistModel(Meijaard2007Model):
    """

    Tdel_total = -kphidot*phidot - kphi*phi + Tdel

    Tdel_total = [kphi, kdelta, kphidot, kdeltadot] * x

    x = [roll angle,
         steer angle,
         roll rate,
         steer rate]

        The inputs are [roll torque,
                        steer torque]

    """

    def form_reduced_canonical_matrices(self, **parameter_overrides):

        par, array_key, array_val = self._parse_parameter_overrides(
            **parameter_overrides)

        if array_val is not None:
            M = np.zeros((len(array_val), 2, 2))
            C1 = np.zeros((len(array_val), 2, 2))
            K0 = np.zeros((len(array_val), 2, 2))
            K2 = np.zeros((len(array_val), 2, 2))
            for i, val in enumerate(array_val):
                par[array_key] = val
                M[i], C1[i], K0[i], K2[i] = benchmark_par_to_canonical(par)
                if array_key == 'kphidot':
                    C1[i, 1, 0] = C1[i, 1, 0] + val
                elif array_key == 'kphi':
                    K0[i, 1, 0] = K0[i, 1, 0] + val
                elif array_key == 'kdeltadot':
                    C1[i, 1, 1] = C1[i, 1, 1] + val
                elif array_key == 'kdelta':
                    K0[i, 1, 1] = K0[i, 1, 1] + val
                else:
                    C1[i, 1, 0] = C1[i, 1, 0] + par['kphidot']
                    K0[i, 1, 0] = K0[i, 1, 0] + par['kphi']
                    C1[i, 1, 1] = C1[i, 1, 1] + par['kdeltadot']
                    K0[i, 1, 1] = K0[i, 1, 1] + par['kdelta']
            return M, C1, K0, K2
        else:
            M, C1, K0, K2 = benchmark_par_to_canonical(par)
            K0[1, 0] = K0[1, 0] + par['kphi']
            K0[1, 1] = K0[1, 1] + par['kdelta']
            C1[1, 0] = C1[1, 0] + par['kphidot']
            C1[1, 1] = C1[1, 1] + par['kdeltadot']
            return M, C1, K0, K2

# NOTE : This is Browser + Jason taken from HumanControl repo
meijaard2007_parameters = {  # dictionary of the parameters in Meijaard 2007
    'IBxx': 11.3557360401,
    'IBxz': -1.96756380745,
    'IByy': 12.2177848012,
    'IBzz': 3.12354397008,
    'IFxx': 0.0904106601579,
    'IFyy': 0.149389340425,
    'IHxx': 0.253379594731,
    'IHxz': -0.0720452391817,
    'IHyy': 0.246138810935,
    'IHzz': 0.0955770796289,
    'IRxx': 0.0883819364527,
    'IRyy': 0.152467620286,
    'c': 0.0685808540382,
    'g': 9.81,
    'lam': 0.399680398707,
    'mB': 81.86,
    'mF': 2.02,
    'mH': 3.22,
    'mR': 3.11,
    'rF': 0.34352982332,
    'rR': 0.340958858855,
    'v': 1.0,
    'w': 1.121,
    'xB': 0.289099434117,
    'xH': 0.866949640247,
    'zB': -1.04029228321,
    'zH': -0.748236400835,
    'kdelta': 0.0,
    'kdeltadot': 0.0,
    'kphi': 0.0,
    'kphidot': 0.0,
}


parameter_set = Meijaard2007ParameterSet(meijaard2007_parameters, True)

model = SteerAssistModel(parameter_set)
cmap = mpl.colormaps['viridis']
gains = [0.0, -1.0, -5.0, -10.0, -50.0, -100.0, -500.0, -1000.0, -5000.0]
color_vals = np.linspace(0.2, 1.0, num=len(gains))

fig, ax = plt.subplots()
for gain, color_val in zip(gains, color_vals):
    color_rgb = cmap(color_val)
    ax = model.plot_eigenvalue_parts(ax=ax, kphidot=gain,
                                     colors=(color_rgb, color_rgb, color_rgb),
                                     v=np.linspace(0.0, 10.0, num=2000))
ax.set_ylim((-10.0, 10.0))

fig.savefig(os.path.join(FIG_DIR, 'roll-rate-eig-effect.png'), dpi=300)

speeds = np.linspace(0.0, 10.0, num=2000)
gain_arrays = {'kphi': [], 'kphidot': [], 'kdelta': [], 'kdeltadot': []}
ctrb_mats = []
R = np.array([1.0])
Q = np.eye(4)
evals = np.zeros((len(speeds), 4), dtype='complex128')
evecs = np.zeros((len(speeds), 4, 4), dtype='complex128')
for i, speed in enumerate(speeds):
    A, B = model.form_state_space_matrices(v=speed)
    B_delta = B[:, 1, np.newaxis]  # shape(4,1)
    C = ct.ctrb(A, B_delta)
    ctrb_mats.append(np.linalg.matrix_rank(C))
    _, evals[i], K = ct.care(A, B_delta, Q, method='slycot')
    K = K.squeeze()
    gain_arrays['kphi'].append(K[0])
    gain_arrays['kdelta'].append(K[1])
    gain_arrays['kphidot'].append(K[2])
    gain_arrays['kdeltadot'].append(K[3])
    A_closed, _ = model.form_state_space_matrices(v=speed,
                                                  kphi=K[0],
                                                  kdelta=K[1],
                                                  kphidot=K[2],
                                                  kdeltadot=K[3])
    evals[i], evecs[i] = np.linalg.eig(A_closed)

fig, ax = plt.subplots()
ax.plot(speeds, np.real(evals), '.k')
ax.set_ylim((-10.0, 10.0))
ax.grid()
fig.savefig(os.path.join(FIG_DIR, 'lqr-eig.png'), dpi=300)

fig, ax = plt.subplots()
ax.plot(ctrb_mats)

#model.plot_eigenvalue_parts(kphidot=np.linspace(-20.0, 10.0, num=1000))
#model.plot_eigenvalue_parts(v=np.linspace(0.0, 10.0, num=1000))
#model.plot_eigenvalue_parts(kphidot=-10.0, v=np.linspace(0.0, 10.0, num=1000))

plt.show()
