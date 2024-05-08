import numpy as np
from numpy import * # For access to all fundamental functions, constants etc.
def Res(z,state,params,Ts):
    """ RES Sequential residual generator for model ''
    Causality: mixed

    Structurally sensitive to faults: f1

    Example of basic usage:
    Let z be the observations matrix, each column corresponding to a known signal and Ts the sampling time,
    then the residual generator can be simulated by:

    r = np.zeros(N) # N number of data points
    state = {'x1': x1_0, 'x3': x3_0}
    for k,zk in enumerate(z):
        r[k], state = Res( zk, state, params, Ts )

    State is a dictionary with the keys: x1, x3

    File generated Wed May  8 11:46:12 2024
    """
    def ApproxInt(dx, x0, Ts):
        return x0 + Ts*dx

    def ApproxDiff(x, xold, Ts):
        return (x - xold) / Ts

    def Res_core(z, state, params, Ts):
        # Known signals
        y1 = z[0]
        y3 = z[1]
        u = z[2]

        # Initialize state variables
        x1 = state['x1']
        x3 = state['x3']

        # Residual generator body
        x3 = y3 # e5
        dx3 = ApproxDiff(x3, state['x3'], Ts) # e8
        x2 = 7.14285714285714*dx3 + 1.42857142857143*x3 # e3
        dx1 = u - 0.2*x1 - 0.1*x2 # e1
         
        r = x1 - y1 # e4

        # Update integrator variables
        x1 = ApproxInt(dx1, state['x1'], Ts) # e6

        # Update state variables
        state['x1'] = x1
        state['x3'] = x3

        return (r, state)

    return Res_core(z, state, params, Ts)
