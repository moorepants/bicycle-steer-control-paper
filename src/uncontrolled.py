import os

from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
import control as ct
import matplotlib as mpl
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

# Plot eigenvalues and eigenvectors of base bicycle
model.plot_eigenvalue_parts(v=np.linspace(0.0, 10.0, num=100))
model.plot_eigenvectors(v=[0.0, 2.5, 5.0, 7.5, 10.0])
