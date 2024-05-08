import numpy as np
from numpy import * # For access to all fundamental functions, constants etc.
def ResGen1_ex2(z,state,params,Ts):
    """ RESGEN1_EX2 Sequential residual generator for model ''
    Causality: int

    Structurally sensitive to faults: f2

    Example of basic usage:
    Let z be the observations matrix, each column corresponding to a known signal and Ts the sampling time,
    then the residual generator can be simulated by:

    r = np.zeros(N) # N number of data points
    state = {'x2': x2_0, 'x4': x4_0}
    for k,zk in enumerate(z):
        r[k], state = ResGen1_ex2( zk, state, params, Ts )

    State is a dictionary with the keys: x2, x4

    File generated Tue May  7 15:49:02 2024
    """
    def ApproxInt(dx, x0, Ts):
        return x0 + Ts*dx

    def ResGen1_ex2_core(z, state, params, Ts):
        # Known signals
        y3 = z[0]
        um = z[2]

        # Initialize state variables
        x2 = state['x2']
        x4 = state['x4']

        # Residual generator body
        u = um # e8
        dx2 = u - 0.2*x2 # e2
        dx4 = 0.1*x2 - 0.2*x4 # e4
        x3 = x4 # e5
         
        r = x3 - y3 # e6

        # Update integrator variables
        x2 = ApproxInt(dx2, state['x2'], Ts) # e10
        x4 = ApproxInt(dx4, state['x4'], Ts) # e12

        # Update state variables
        state['x2'] = x2
        state['x4'] = x4

        return (r, state)

    return ResGen1_ex2_core(z, state, params, Ts)
