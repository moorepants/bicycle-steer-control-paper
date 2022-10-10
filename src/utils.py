import control as ct
import numpy as np
import scipy.optimize as spo


def calc_det_control(model, speed):
    A, B = model.form_state_space_matrices(v=float(speed))
    # only steer control
    B_delta = B[:, 1, np.newaxis]  # shape(4,1)
    C = ct.ctrb(A, B_delta)
    return np.linalg.det(C)


def find_uncontrollable_points(model):
    uncontrollable_points = []
    # NOTE : If I make the first v 0.0, it finds 0.0 to be a root, but it
    # clearly isn't if you check that speed. If you select it smaller than 1e-7
    # fsolve doesn't converged.
    for v in np.linspace(1e-7, 10.0, num=20):
        res = spo.fsolve(lambda sp: calc_det_control(model, sp), v, xtol=1e-12)
        uncontrollable_points.append(res[0])
    uncontrollable_points = np.array(uncontrollable_points)
    un, idxs = np.unique(np.round(uncontrollable_points, 10),
                         return_index=True)
    return uncontrollable_points[idxs]
