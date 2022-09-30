import os

from scipy.linalg import solve_continuous_are
import control as ct
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from bicycleparameters.bicycle import benchmark_par_to_canonical, ab_matrix
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

    def form_state_space_matrices(self, **parameter_overrides):
        """Returns the A and B matrices for the Whipple model linearized about
        the upright constant velocity configuration.

        Parameters
        ==========
        speed : float
            The speed of the bicycle.

        Returns
        =======
        A : ndarray, shape(4,4)
            The state matrix.
        B : ndarray, shape(4,2)
            The input matrix.

        Notes
        =====
        ``A`` and ``B`` describe the Whipple model in state space form:

            x' = A * x + B * u

        where

        The states are [roll angle,
                        steer angle,
                        roll rate,
                        steer rate]

        The inputs are [roll torque,
                        steer torque]

        """
        # These parameters are not used in the computation of M, C1, K0, K2.
        gain_names = ['kphi', 'kdelta', 'kphidot', 'kdeltadot']
        non_canon = {}
        for par_name in ['g', 'v'] + gain_names:
            if par_name in parameter_overrides.keys():
                non_canon[par_name] = parameter_overrides[par_name]
            else:
                non_canon[par_name] = self.parameter_set.parameters[par_name]

        M, C1, K0, K2 = self.form_reduced_canonical_matrices(
            **parameter_overrides)

        # steer controller gains
        K = np.array([[0.0, 0.0, 0.0, 0.0],
                      [non_canon[p] for p in gain_names]])

        if len(M.shape) == 3:  # one of the canonical parameters is an array
            A = np.zeros((M.shape[0], 4, 4))
            B = np.zeros((M.shape[0], 4, 2))
            for i, (Mi, C1i, K0i, K2i) in enumerate(zip(M, C1, K0, K2)):
                Ai, Bi = ab_matrix(Mi, C1i, K0i, K2i, non_canon['v'],
                                   non_canon['g'])
                A[i] = Ai - Bi@K
                B[i] = Bi
        elif not isinstance(non_canon['v'], float):
            A = np.zeros((len(non_canon['v']), 4, 4))
            B = np.zeros((len(non_canon['v']), 4, 2))
            for i, vi in enumerate(non_canon['v']):
                Ai, Bi = ab_matrix(M, C1, K0, K2, vi, non_canon['g'])
                A[i] = Ai - Bi@K
                B[i] = Bi
        elif not isinstance(non_canon['g'], float):
            A = np.zeros((len(non_canon['g']), 4, 4))
            B = np.zeros((len(non_canon['g']), 4, 2))
            for i, gi in enumerate(non_canon['g']):
                Ai, Bi = ab_matrix(M, C1, K0, K2, non_canon['v'], gi)
                A[i] = Ai - Bi@K
                B[i] = Bi
        else:  # scalar parameters
            A, B = ab_matrix(M, C1, K0, K2, non_canon['v'], non_canon['g'])
            A = A - B@K
            B = B
        # TODO : implement if one of the gains is an array

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
#Q = np.diag([1.0, 0.5, 10.0, 0.5])
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
    #evals[i], evecs[i] = np.linalg.eig(A_closed)

fig, ax = plt.subplots()
ax.plot(speeds, np.real(evals), '.k')
ax.plot(speeds, np.imag(evals), '.b')
ax.set_ylim((-10.0, 10.0))
ax.grid()
fig.savefig(os.path.join(FIG_DIR, 'lqr-eig.png'), dpi=300)

fig, ax = plt.subplots()
ax.plot(ctrb_mats)

#model.plot_eigenvalue_parts(kphidot=np.linspace(-20.0, 10.0, num=1000))
#model.plot_eigenvalue_parts(v=np.linspace(0.0, 10.0, num=1000))
#model.plot_eigenvalue_parts(kphidot=-10.0, v=np.linspace(0.0, 10.0, num=1000))

plt.show()
