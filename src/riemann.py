"""Module containing all Riemann solvers."""
import numpy as np
from pycfd_types import real_t
from states import State, primToCons
from physics import speed_of_sound
import params
from varindexes import IR, IU, IV, IP, IE
# IR, IU, IV, IW, IP, IE, IBX, IBY, IBZ, IPSI = VarIndex.__members__


def logMean(a: float, b: float) -> float:
    """
    Compute the logarithmic mean of two numbers.
    """
    if a == b:
        return a
    return (b - a) / (np.log(b) - np.log(a))

# def FluxKepec(qL: State, qR: State, ch: float) -> State:
#     qAvg = (qL + qR) * 0.5
#     rhoLn = logMean(qL.r, qR.r)
#     betaL = 0.5 * (qL.r / qL.p)
#     betaR = 0.5 * (qR.r / qR.p)
#     betaAvg = 0.5 * qAvg.r / qAvg.p
#     betaLn = 0.5 * rhoLn / logMean(qL.p, qR.p)
#     B2Avg = (qL.bx**2 + qR.bx**2)/2 + (qL.by**2 + qR.by**2)/2 + (qL.bz**2 + qR.bz**2)/2
#     f1 = rhoLn * qAvg.u
#     f2 = f1 * qAvg.u - qAvg.bx * qAvg.bx + 0.5 * qAvg.r/betaAvg + 0.5 * B2Avg
#     f3 = f1 * qAvg.v - qAvg.by * qAvg.bx
#     f4 = f1 * qAvg.w - qAvg.bz * qAvg.bx
#     f6 = ch *qAvg.bx
#     f7 = 1/betaAvg * ((qL.u*betaL + qR.u*betaR/2)*qAvg.by - (qL.v*betaL + qR.v*betaR/2)*qAvg.bx)
#     f8 = 1/betaAvg * ((qL.u*betaL + qR.u*betaR/2)*qAvg.bz - (qL.w*betaL + qR.w*betaR/2)*qAvg.bx)
#     f9 = ch * qAvg.bx
#     f5 = f1 * (1/(2*betaLn*(params.gamma - 1)) - 0.5 * (qL.u*qR.u + qL.v*qR.v + qL.w*qR.w)) + \
#          f2 * qAvg.u + \
#          f3 * qAvg.v + \
#          f4 * qAvg.w + \
#          f6 * qAvg.bx + \
#          f7 * qAvg.by + \
#          f8 * qAvg.bz + \
#          f9 * qAvg.psi + \
#          qAvg.bx * (qAvg.u*qAvg.bx + qAvg.v*qAvg.by + qAvg.w*qAvg.bz) - 0.5 * qAvg.u * B2Avg - ch * qAvg.psi * qAvg.bx
    
#     return State(
#         f1,
#         f2,
#         f3,
#         f4,
#         f5,
#         f6,
#         f7,
#         f8,
#         f9
#     )


# def IdealGLM(qL: State, qR: State) -> State:
#     flux = FluxKepec(qL, qR)
#     return flux

def computeFlux(q: State) -> State:
    Ek: real_t = 0.5 * q[IR] * (q[IU] * q[IU] + q[IV] * q[IV])
    E: real_t = (q[IP] / (params.gamma-1.0) + Ek)

    fout = State()

    fout[IR] = q[IR]*q[IU]
    fout[IU] = q[IR]*q[IU]*q[IU] + q[IP]
    fout[IV] = q[IR]*q[IU]*q[IV]
    fout[IE] = (q[IP] + E) * q[IU]
    return fout


def hll(qL: State, qR: State) -> State:
    aL: float = speed_of_sound(qL)
    aR: float = speed_of_sound(qR)

  # Davis' estimates for the signal speed
    sminL: float = qL[IU] - aL
    smaxL: float = qL[IU] + aL
    sminR: float = qR[IU] - aR
    smaxR: float = qR[IU] + aR

    SL: real_t = min(sminL, sminR)
    SR: real_t = max(smaxL, smaxR)

    FL: State = computeFlux(qL)
    FR: State = computeFlux(qR)
    flux: State
    if (SL >= 0.0):
      flux = FL
      # pout = qL[IP]
    elif (SR <= 0.0):
      flux = FR
      # pout = qR[IP]
    else:
        uL: State = primToCons(qL)
        uR: State = primToCons(qR)
        # pout: real_t = 0.5 * (qL[IP] + qR[IP]);
        flux = (SR*FL - SL*FR + SL*SR*(uR-uL)) / (SR-SL)
    return flux

# Calling the right Riemann solver
def riemann(qL: State, qR: State) -> State:
  match (params.riemann_solver):
     case "HLL":
        flux = hll(qL, qR)
    # case "HLLC": hllc(qL, qR, flux, pout, params); break;
  return flux

