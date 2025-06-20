"""Module containing all Riemann solvers."""
import numpy as np
from states import State

gamma = 5/3

def logMean(a: float, b: float) -> float:
    """
    Compute the logarithmic mean of two numbers.
    """
    if a == b:
        return a
    return (b - a) / (np.log(b) - np.log(a))

def FluxKepec(qL: State, qR: State, ch: float) -> State:
    qAvg = (qL + qR) * 0.5
    rhoLn = logMean(qL.r, qR.r)
    betaL = 0.5 * (qL.r / qL.p)
    betaR = 0.5 * (qR.r / qR.p)
    betaAvg = 0.5 * qAvg.r / qAvg.p
    betaLn = 0.5 * rhoLn / logMean(qL.p, qR.p)
    B2Avg = (qL.bx**2 + qR.bx**2)/2 + (qL.by**2 + qR.by**2)/2 + (qL.bz**2 + qR.bz**2)/2
    f1 = rhoLn * qAvg.u
    f2 = f1 * qAvg.u - qAvg.bx * qAvg.bx + 0.5 * qAvg.r/betaAvg + 0.5 * B2Avg
    f3 = f1 * qAvg.v - qAvg.by * qAvg.bx
    f4 = f1 * qAvg.w - qAvg.bz * qAvg.bx
    f6 = ch *qAvg.bx
    f7 = 1/betaAvg * ((qL.u*betaL + qR.u*betaR/2)*qAvg.by - (qL.v*betaL + qR.v*betaR/2)*qAvg.bx)
    f8 = 1/betaAvg * ((qL.u*betaL + qR.u*betaR/2)*qAvg.bz - (qL.w*betaL + qR.w*betaR/2)*qAvg.bx)
    f9 = ch * qAvg.bx
    f5 = f1 * (1/(2*betaLn*(gamma - 1)) - 0.5 * (qL.u*qR.u + qL.v*qR.v + qL.w*qR.w)) + \
         f2 * qAvg.u + \
         f3 * qAvg.v + \
         f4 * qAvg.w + \
         f6 * qAvg.bx + \
         f7 * qAvg.by + \
         f8 * qAvg.bz + \
         f9 * qAvg.psi + \
         qAvg.bx * (qAvg.u*qAvg.bx + qAvg.v*qAvg.by + qAvg.w*qAvg.bz) - 0.5 * qAvg.u * B2Avg - ch * qAvg.psi * qAvg.bx
    
    return State(
        f1,
        f2,
        f3,
        f4,
        f5,
        f6,
        f7,
        f8,
        f9
    )


def IdealGLM(qL: State, qR: State) -> State:
    flux = FluxKepec(qL, qR)
    return flux