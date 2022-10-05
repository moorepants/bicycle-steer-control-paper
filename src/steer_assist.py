import os

import control as ct
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from bicycleparameters.bicycle import ab_matrix
from bicycleparameters.models import Meijaard2007Model
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet

SCRIPT_PATH = os.path.realpath(__file__)
SRC_DIR = os.path.dirname(SCRIPT_PATH)
ROOT_DIR = os.path.realpath(os.path.join(SRC_DIR, '..'))
FIG_DIR = os.path.join(ROOT_DIR, 'figures')

if not os.path.exists(FIG_DIR):
    os.mkdir(FIG_DIR)


class SteerControlModel(Meijaard2007Model):
    """

    Tdel_total = -kphidot*phidot - kphi*phi + Tdel

    Tdel_total = -[kphi, kdelta, kphidot, kdeltadot] * x

    x = [roll angle,
         steer angle,
         roll rate,
         steer rate]

        The inputs are [roll torque,
                        steer torque]

    """

    def form_state_space_matrices(self, **parameter_overrides):
        """Returns the A and B matrices for the Whipple-Carvallo model
        linearized about the upright constant velocity configuration with a
        full state feedback steer controller.

        Returns
        =======
        A : ndarray, shape(4,4) or shape(n,4,4)
            The state matrix.
        B : ndarray, shape(4,2) or shape(n,4,2)
            The input matrix.

        Notes
        =====
        A, B, and K describe the model in state space form:

            x' = (A - B*K)*x + B*u

        where::

        x = |phi     | = |roll angle |
            |delta   |   |steer angle|
            |phidot  |   |roll rate  |
            |deltadot|   |steer rate |

        K = |0    0      0       0        |
            |kphi kdelta kphidot kdeltadot|

        u = |Tphi  | = |roll torque |
            |Tdelta|   |steer torque|

        """
        gain_names = ['kphi', 'kdelta', 'kphidot', 'kdeltadot']

        par, array_keys, array_len = self._parse_parameter_overrides(
            **parameter_overrides)

        # g, v, and the contoller gains are not used in the computation of M,
        # C1, K0, K2.

        M, C1, K0, K2 = self.form_reduced_canonical_matrices(
            **parameter_overrides)

        # steer controller gains, 2x4, no roll control
        if any(k in gain_names for k in array_keys):
            # if one of the gains is an array, create a set of gain matrices
            # where that single gain varies across the set
            K = np.array([[0.0, 0.0, 0.0, 0.0],
                          [par[p][0] if p in array_keys else par[p]
                           for p in gain_names]])
            # K is now shape(n, 2, 4)
            K = np.tile(K, (array_len, 1, 1))
            for k in array_keys:
                if k in gain_names:
                    K[:, 1, gain_names.index(k)] = par[k]
        else:  # gains are not an array
            K = np.array([[0.0, 0.0, 0.0, 0.0],
                          [par[p] for p in gain_names]])

        if array_keys:
            A = np.zeros((array_len, 4, 4))
            B = np.zeros((array_len, 4, 2))
            for i in range(array_len):
                Mi = M[i] if M.ndim == 3 else M
                C1i = C1[i] if C1.ndim == 3 else C1
                K0i = K0[i] if K0.ndim == 3 else K0
                K2i = K2[i] if K2.ndim == 3 else K2
                vi = par['v'] if np.isscalar(par['v']) else par['v'][i]
                gi = par['g'] if np.isscalar(par['g']) else par['g'][i]
                Ki = K[i] if K.ndim == 3 else K
                Ai, Bi = ab_matrix(Mi, C1i, K0i, K2i, vi, gi)
                A[i] = Ai - Bi@Ki
                B[i] = Bi
        else:  # scalar parameters
            A, B = ab_matrix(M, C1, K0, K2, par['v'], par['g'])
            A = A - B@K
            B = B

        return A, B


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
model = SteerControlModel(parameter_set)

# Figure 1: Plot the real & imaginary eigenvalue parts as a function of speed
# for a series of roll derivative control gains.
cmap = mpl.colormaps['viridis']
gains = [0.0, -1.0, -5.0, -10.0, -50.0, -100.0, -500.0]  # , -1000.0, -5000.0]
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
                               v=[0.0, 3.0, 5.0, 7.0, 9.0])
fig = axes[0, 0].figure
fig.savefig(os.path.join(FIG_DIR, 'roll-rate-gain-evec-effect.png'),
            dpi=300)


speeds = np.linspace(0.0, 10.0, num=1000)
gain_arrays = {'kphi': [], 'kphidot': [], 'kdelta': [], 'kdeltadot': []}
ctrb_mats = []
R = np.array([1.0])
Q = np.eye(4)
#Q = np.diag([1.0, 0.5, 10.0, 0.5])
evals = np.zeros((len(speeds), 4), dtype='complex128')
evecs = np.zeros((len(speeds), 4, 4), dtype='complex128')
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
    A_closed, _ = model.form_state_space_matrices(v=speed,
                                                  kphi=K[0],
                                                  kdelta=K[1],
                                                  kphidot=K[2],
                                                  kdeltadot=K[3])
    evals[i], evecs[i] = np.linalg.eig(A_closed)

fig, ax = plt.subplots()
ax = model.plot_eigenvalue_parts(ax=ax, v=speeds)
ax.plot(speeds, np.real(evals), '.k')
ax.plot(speeds, np.imag(evals), '.b')
ax.set_ylim((-10.0, 10.0))
ax.grid()
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
