import os

from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
import control as ct
import control.optimal as ctop
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

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

A, B = model.form_state_space_matrices(v=1.0)
C = np.eye(4)
T = 0.1

#https://math.stackexchange.com/questions/2157745/state-space-discretization
Ad = scipy.linalg.expm(A*T)
Bd = np.linalg.solve(A, (Ad - np.eye(4))@B)

iosys = ct.ss2io(ct.ss(Ad, Bd[:, 1, np.newaxis], C, 0, T))

ud = np.array([3.8])
xd = np.linalg.solve(np.eye(4) - Ad, Bd[:, 1, np.newaxis] @ ud)

constraints = [ctop.input_range_constraint(iosys, [-10], [10])]

Q = np.eye(4)
R = np.eye(1)

cost = ctop.quadratic_cost(iosys, Q, R, x0=xd, u0=ud)

cont = ctop.create_mpc_iosystem(iosys, np.arange(0, 20)*T, cost, constraints)

loop = ct.feedback(iosys, cont, 1)

resp = ct.input_output_response(loop, np.arange(0, 100)*T, 0, 0)
plt.plot(resp.time, resp.outputs.T)
