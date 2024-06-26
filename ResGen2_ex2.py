import numpy as np
from numpy import * # For access to all fundamental functions, constants etc.
def ResGen2_ex2(z,state,params,Ts):
    """ RESGEN2_EX2 Sequential residual generator for model ''
    Causality: int

    Structurally sensitive to faults: f1

    Example of basic usage:
    Let z be the observations matrix, each column corresponding to a known signal and Ts the sampling time,
    then the residual generator can be simulated by:

    r = np.zeros(N) # N number of data points
    state = {'x1': x1_0, 'x3': x3_0}
    for k,zk in enumerate(z):
        r[k], state = ResGen2_ex2( zk, state, params, Ts )

    State is a dictionary with the keys: x1, x3

    File generated Tue May  7 15:49:02 2024
    """
    def ApproxInt(dx, x0, Ts):
        return x0 + Ts*dx

    def ResGen2_ex2_core(z, state, params, Ts):
        # Known signals
        y3 = z[0]
        um = z[2]

        # Initialize state variables
        x1 = state['x1']
        x3 = state['x3']

        # Residual generator body
        u = um # e8
        dx1 = u - 0.2*x1 # e1
        dx3 = 0.1*x1 - 0.2*x3 # e3
         
        r = x3 - y3 # e6

        # Update integrator variables
        x1 = ApproxInt(dx1, state['x1'], Ts) # e9
        x3 = ApproxInt(dx3, state['x3'], Ts) # e11

        # Update state variables
        state['x1'] = x1
        state['x3'] = x3

        return (r, state)

    return ResGen2_ex2_core(z, state, params, Ts)
