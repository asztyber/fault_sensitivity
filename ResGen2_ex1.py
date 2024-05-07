import numpy as np
from numpy import * # For access to all fundamental functions, constants etc.
def ResGen2_ex1(z,state,params,Ts):
    """ RESGEN2_EX1 Sequential residual generator for model ''
    Causality: int

    Structurally sensitive to faults: f1, f2

    Example of basic usage:
    Let z be the observations matrix, each column corresponding to a known signal and Ts the sampling time,
    then the residual generator can be simulated by:

    r = np.zeros(N) # N number of data points
    state = {'x1': x1_0, 'x2': x2_0}
    for k,zk in enumerate(z):
        r[k], state = ResGen2_ex1( zk, state, params, Ts )

    State is a dictionary with the keys: x1, x2

    File generated Tue May  7 10:56:55 2024
    """
    def ApproxInt(dx, x0, Ts):
        return x0 + Ts*dx

    def ResGen2_ex1_core(z, state, params, Ts):
        # Known signals
        y1 = z[0]
        y3 = z[1]
        u = z[2]

        # Initialize state variables
        x1 = state['x1']
        x2 = state['x2']

        # Residual generator body
        x3 = y3 # e5
        dx2 = 0.1*x1 - 0.2*x2 - 0.1*x3 # e2
        dx1 = u - 0.2*x1 - 0.1*x2 # e1
         
        r = x1 - y1 # e4

        # Update integrator variables
        x1 = ApproxInt(dx1, state['x1'], Ts) # e6
        x2 = ApproxInt(dx2, state['x2'], Ts) # e7

        # Update state variables
        state['x1'] = x1
        state['x2'] = x2

        return (r, state)

    return ResGen2_ex1_core(z, state, params, Ts)
