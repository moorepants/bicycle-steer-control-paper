import sympy as sm

states = [
    sm.symbols('phi'),
    sm.symbols('delta'),
    sm.symbols('phidot'),
    sm.symbols('deltadot'),
]

inputs = [
    sm.symbols('Tphi'),
    sm.symbols('Tdelta'),
]

constants = {
    'IBxx': sm.symbols('IBxx'),
    'IBxz': sm.symbols('IBxz'),
    'IByy': sm.symbols('IByy'),
    'IBzz': sm.symbols('IBzz'),
    'IFxx': sm.symbols('IFxx'),
    'IFyy': sm.symbols('IFyy'),
    'IHxx': sm.symbols('IHxx'),
    'IHxz': sm.symbols('IHxz'),
    'IHyy': sm.symbols('IHyy'),
    'IHzz': sm.symbols('IHzz'),
    'IRxx': sm.symbols('IRxx'),
    'IRyy': sm.symbols('IRyy'),
    'c': sm.symbols('c'),
    'g': sm.symbols('g'),
    'lam': sm.symbols('lam'),
    'mB': sm.symbols('mB'),
    'mF': sm.symbols('mF'),
    'mH': sm.symbols('mH'),
    'mR': sm.symbols('mR'),
    'rF': sm.symbols('rF'),
    'rR': sm.symbols('rR'),
    'v': sm.symbols('v'),
    'w': sm.symbols('w'),
    'xB': sm.symbols('xB'),
    'xH': sm.symbols('xH'),
    'zB': sm.symbols('zB'),
    'zH': sm.symbols('zH'),
    'kdelta': sm.symbols('kdelta'),
    'kdeltadot': sm.symbols('kdeltadot'),
    'kphi': sm.symbols('kphi'),
    'kphidot': sm.symbols('kphidot'),
}


def benchmark_par_to_canonical(p):
    '''Returns the canonical matrices of the Whipple bicycle model linearized
    about the upright constant velocity configuration. It uses the parameter
    definitions from Meijaard et al. 2007.

    Parameters
    ----------
    p : dictionary
        A dictionary of the benchmark bicycle parameters. Make sure your units
        are correct, best to ue the benchmark paper's units!

    Returns
    -------
    M : ndarray, shape(2,2)
        The mass matrix.
    C1 : ndarray, shape(2,2)
        The damping like matrix that is proportional to the speed, v.
    K0 : ndarray, shape(2,2)
        The stiffness matrix proportional to gravity, g.
    K2 : ndarray, shape(2,2)
        The stiffness matrix proportional to the speed squared, v**2.

    Notes
    -----
    This function handles parameters with uncertanties.

    '''
    mT = p['mR'] + p['mB'] + p['mH'] + p['mF']
    xT = (p['xB'] * p['mB'] + p['xH'] * p['mH'] + p['w'] * p['mF']) / mT
    zT = (-p['rR'] * p['mR'] + p['zB'] * p['mB'] +
          p['zH'] * p['mH'] - p['rF'] * p['mF']) / mT

    ITxx = (p['IRxx'] + p['IBxx'] + p['IHxx'] + p['IFxx'] + p['mR'] *
            p['rR']**2 + p['mB'] * p['zB']**2 + p['mH'] * p['zH']**2 + p['mF']
            * p['rF']**2)
    ITxz = (p['IBxz'] + p['IHxz'] - p['mB'] * p['xB'] * p['zB'] -
            p['mH'] * p['xH'] * p['zH'] + p['mF'] * p['w'] * p['rF'])
    # p['IRzz'] = p['IRxx']
    # p['IFzz'] = p['IFxx']
    ITzz = (p['IRxx'] + p['IBzz'] + p['IHzz'] + p['IFxx'] +
            p['mB'] * p['xB']**2 + p['mH'] * p['xH']**2 + p['mF'] * p['w']**2)

    mA = p['mH'] + p['mF']
    xA = (p['xH'] * p['mH'] + p['w'] * p['mF']) / mA
    zA = (p['zH'] * p['mH'] - p['rF']* p['mF']) / mA

    IAxx = (p['IHxx'] + p['IFxx'] + p['mH'] * (p['zH'] - zA)**2 +
            p['mF'] * (p['rF'] + zA)**2)
    IAxz = (p['IHxz'] - p['mH'] * (p['xH'] - xA) * (p['zH'] - zA) + p['mF'] *
            (p['w'] - xA) * (p['rF'] + zA))
    IAzz = (p['IHzz'] + p['IFxx'] + p['mH'] * (p['xH'] - xA)**2 + p['mF'] *
            (p['w'] - xA)**2)
    uA = (xA - p['w'] - p['c']) * sm.cos(p['lam']) - zA * sm.sin(p['lam'])
    IAll = (mA * uA**2 + IAxx * sm.sin(p['lam'])**2 +
            2 * IAxz * sm.sin(p['lam']) * sm.cos(p['lam']) +
            IAzz * sm.cos(p['lam'])**2)
    IAlx = (-mA * uA * zA + IAxx * sm.sin(p['lam']) + IAxz *
            sm.cos(p['lam']))
    IAlz = (mA * uA * xA + IAxz * sm.sin(p['lam']) + IAzz *
            sm.cos(p['lam']))

    mu = p['c'] / p['w'] * sm.cos(p['lam'])

    SR = p['IRyy'] / p['rR']
    SF = p['IFyy'] / p['rF']
    ST = SR + SF
    SA = mA * uA + mu * mT * xT

    Mpp = ITxx
    Mpd = IAlx + mu * ITxz
    Mdp = Mpd
    Mdd = IAll + 2 * mu * IAlz + mu**2 * ITzz
    M = sm.Matrix([[Mpp, Mpd], [Mdp, Mdd]])

    K0pp = mT * zT
    K0pd = -SA
    K0dp = K0pd
    K0dd = -SA * sm.sin(p['lam'])
    K0 = sm.Matrix([[K0pp, K0pd], [K0dp, K0dd]])

    K2pp = 0.
    K2pd = (ST - mT * zT) / p['w'] * sm.cos(p['lam'])
    K2dp = 0.
    K2dd = (SA + SF * sm.sin(p['lam'])) / p['w'] * sm.cos(p['lam'])
    K2 = sm.Matrix([[K2pp, K2pd], [K2dp, K2dd]])

    C1pp = 0.
    C1pd = (mu*ST + SF*sm.cos(p['lam']) + ITxz / p['w'] *
            sm.cos(p['lam']) - mu*mT*zT)
    C1dp = -(mu * ST + SF * sm.cos(p['lam']))
    C1dd = (IAlz / p['w'] * sm.cos(p['lam']) + mu * (SA +
            ITzz / p['w'] * sm.cos(p['lam'])))
    C1 = sm.Matrix([[C1pp, C1pd], [C1dp, C1dd]])

    return M, C1, K0, K2


def ab_matrix(M, C1, K0, K2, p):
    '''Calculate the A and B matrices for the Whipple bicycle model linearized
    about the upright configuration.

    Parameters
    ----------
    M : ndarray, shape(2,2)
        The mass matrix.
    C1 : ndarray, shape(2,2)
        The damping like matrix that is proportional to the speed, v.
    K0 : ndarray, shape(2,2)
        The stiffness matrix proportional to gravity, g.
    K2 : ndarray, shape(2,2)
        The stiffness matrix proportional to the speed squared, v**2.
    v : float
        Forward speed.
    g : float
        Acceleration due to gravity.

    Returns
    -------
    A : ndarray, shape(4,4)
        State matrix.
    B : ndarray, shape(4,2)
        Input matrix.

    The states are [roll angle,
                    steer angle,
                    roll rate,
                    steer rate]
    The inputs are [roll torque,
                    steer torque]

    '''

    invM = (1. / (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) *
           sm.Matrix([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]))

    a11 = sm.zeros(2, 2)
    a12 = sm.eye(2)
    # stiffness based terms
    a21 = -invM @ (p['g'] * K0 + p['v']**2 * K2)
    # damping based terms
    a22 = -invM @ (p['v'] * C1)

    #A = np.vstack((np.hstack((a11, a12)), np.hstack((a21, a22))))

    A = a11.row_join(a12).col_join(a21.row_join(a22))

    #B = np.vstack((np.zeros((2, 2)), invM))

    B = a11.col_join(invM)

    K = sm.Matrix([[0, 0, 0, 0],
                   [p['kphi'], p['kdelta'], p['kphidot'], p['kdeltadot']]])

    return A, B, K

M, C1, K0, K2 = benchmark_par_to_canonical(constants)
A, B, K = ab_matrix(M, C1, K0, K2, constants)
