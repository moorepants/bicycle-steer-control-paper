"""Trajectory optimization using the linear bicycle model."""

import numpy as np
import pycollo
import sympy as sm

from data import bike_with_rider
from sym_model import states, inputs, constants, A, B, K, M, C1, K0, K2


problem = pycollo.OptimalControlProblem(name='Linear Bicycle Steer Control')
phase = problem.new_phase(name='A')

phi, delta, phidot, deltadot = states
phiddot, deltaddot = sm.symbols('phiddot deltaddot')
Tphi, Tdelta = inputs

phase.state_variables = [phi, delta, phidot, deltadot]
phase.control_variables = [Tdelta, phiddot, deltaddot]
phase.state_equations = {
    phi: phidot,
    delta: deltadot,
    phidot: phiddot,
    deltadot: deltaddot,
}
# yddot = (A - B@K)@sm.Matrix(states) + B@sm.Matrix(inputs)
ydot = (A - B@K)@sm.Matrix(states) + B@sm.Matrix(inputs)
phase.path_constraints = [ydot[2, 0] - phiddot, ydot[3, 0] - deltaddot]
phase.integrand_functions = [Tdelta**2]

phase.bounds.initial_time = 0.0
phase.bounds.final_time = [1.0, 10.0]
phase.bounds.state_variables = {
    phi: [-np.pi/2, np.pi/2],
    delta: [-np.pi/2, np.pi/2],
    phidot: [-np.deg2rad(200), np.deg2rad(200)],
    deltadot: [-np.deg2rad(200), np.deg2rad(200)],
}
phase.bounds.control_variables = {
    Tdelta: [-1000, 1000],
    phiddot: [-1000, 1000],
    deltaddot: [-1000, 1000],
}
phase.bounds.initial_state_constraints = {
    phi: np.deg2rad(10),
    delta: np.deg2rad(20),
    phidot: 0,
    deltadot: 0,
}
phase.bounds.final_state_constraints = {
    phi: [-np.deg2rad(0.1), np.deg2rad(0.1)],
    delta: [-np.deg2rad(0.1), np.deg2rad(0.1)],
    phidot: [-np.deg2rad(0.1), np.deg2rad(0.1)],
    deltadot: [-np.deg2rad(0.1), np.deg2rad(0.1)],
}
phase.bounds.path_constraints = [[0, 0], [0, 0]]
phase.bounds.integral_variables = [0, 100000]

phase.guess.time = np.array([0.0, 10.0])
phase.guess.state_variables = np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
])
phase.guess.control_variables  = np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
])
phase.guess.integral_variables = np.array([0.0])

# phase.mesh.number_mesh_sections = 5

bike_with_rider_constants = {}
for k, v in constants.items():
    symbol = v
    value = bike_with_rider[k]
    bike_with_rider_constants.update({symbol: value})
problem.auxiliary_data = {
    Tphi: 0,
    **bike_with_rider_constants,
}
problem.auxiliary_data[sm.symbols('v')] = 10.0
# problem.auxiliary_data[sm.symbols('kphi')] = -182.855
# problem.auxiliary_data[sm.symbols('kdelta')] = 11.854
# problem.auxiliary_data[sm.symbols('kphidot')] = -63.535
# problem.auxiliary_data[sm.symbols('kdeltadot')] = 2.324

problem.objective_function = phase.integral_variables[0]

problem.settings.nlp_tolerance = 1e-8
problem.settings.mesh_tolerance = 1e-6
problem.settings.max_mesh_iterations = 1
problem.settings.display_mesh_result_graph = True

problem.solve()
