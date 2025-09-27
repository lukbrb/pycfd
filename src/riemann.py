"""Module containing all Riemann solvers."""

import numpy as np
from src.pycfd_types import real_t, Array, IDir
from src.states import State, primToCons
from src.physics import speed_of_sound
import src.params as params
from src.varindexes import IR, IU, IV, IW, IP, IE, IBX, IBY, IBZ

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
    E: real_t = q[IP] / (params.gamma - 1.0) + Ek

    fout = State()

    fout[IR] = q[IR] * q[IU]
    fout[IU] = q[IR] * q[IU] * q[IU] + q[IP]
    fout[IV] = q[IR] * q[IU] * q[IV]
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
    flux: State = State()
    if SL >= 0.0:
        flux = FL
    # pout = qL[IP]
    elif SR <= 0.0:
        flux = FR
    # pout = qR[IP]
    else:
        uL: State = primToCons(qL)  # type: ignore
        uR: State = primToCons(qR)  # type: ignore
        # pout: real_t = 0.5 * (qL[IP] + qR[IP]);
        flux = (SR * FL - SL * FR + SL * SR * (uR - uL)) / (SR - SL)
    return flux


def fivewaves(qL: State, qR: State) -> State:
    B2L: real_t = qL[IBX] * qL[IBX] + qL[IBY] * qL[IBY] + qL[IBZ] * qL[IBZ]
    B2R: real_t = qR[IBX] * qR[IBX] + qR[IBY] * qR[IBY] + qR[IBZ] * qR[IBZ]
    pL: Array = np.array(
        [-qL[IBX] * qL[IBX] + qL[IP] + B2L / 2, -qL[IBX] * qL[IBY], -qL[IBX] * qL[IBZ]]
    )

    pR = np.array(
        [-qR[IBX] * qR[IBX] + qR[IP] + B2R / 2, -qR[IBX] * qR[IBY], -qR[IBX] * qR[IBZ]]
    )

    # 1. Compute speeds
    csL: real_t = speed_of_sound(qL)
    csR: real_t = speed_of_sound(qR)
    caL: real_t = np.sqrt(qL[IR] * (qL[IBX] * qL[IBX] + B2L / 2)) + params.epsilon
    caR: real_t = np.sqrt(qR[IR] * (qR[IBX] * qR[IBX] + B2R / 2)) + params.epsilon
    cbL: real_t = np.sqrt(
        qL[IR] * (qL[IR] * csL * csL + qL[IBY] * qL[IBY] + qL[IBZ] * qL[IBZ] + B2L / 2)
    )
    cbR: real_t = np.sqrt(
        qR[IR] * (qR[IR] * csR * csR + qR[IBY] * qR[IBY] + qR[IBZ] * qR[IBZ] + B2R / 2)
    )

    def computeFastMagnetoAcousticSpeed(q: State, B2: real_t, cs: real_t) -> real_t:
        c02: real_t = cs * cs
        ca2: real_t = B2 / q[IR]
        cap2: real_t = q[IBX] * q[IBX] / q[IR]
        return np.sqrt(
            0.5 * (c02 + ca2)
            + 0.5 * np.sqrt((c02 + ca2) * (c02 + ca2) - 4.0 * c02 * cap2)
        )

    # Using 3-wave if hyperbolicity is lost (from Dyablo)
    if (
        qL[IBX] * qR[IBX] < -params.epsilon
        or qL[IBY] * qR[IBY] < -params.epsilon
        or qL[IBZ] * qR[IBZ] < -params.epsilon
    ):
        clocL = qL[IR] * computeFastMagnetoAcousticSpeed(qL, B2L, csL)
        clocR = qR[IR] * computeFastMagnetoAcousticSpeed(qR, B2R, csR)
        c = max(clocL, clocR)

        caL = c
        caR = c
        cbL = c
        cbR = c

    cL: Array = np.array([cbL, caL, caL])
    cR: Array = np.array([cbR, caR, caR])

    # 2. Compute star zone
    vL: Array = np.array([qL[IU], qL[IV], qL[IW]])
    vR: Array = np.array([qR[IU], qR[IV], qR[IW]])

    Ustar = np.zeros(3)
    Pstar = np.zeros(3)

    for i in range(3):
        Ustar[i] = (cL[i] * vL[i] + cR[i] * vR[i] + pL[i] - pR[i]) / (cL[i] + cR[i])
        Pstar[i] = (cR[i] * pL[i] + cL[i] * pR[i] + cL[i] * cR[i] * (vL[i] - vR[i])) / (
            cL[i] + cR[i]
        )

    if Ustar[IDir.IX] > 0.0:
        q = qL
        Bstar = qR[IBX]
        # pout = qR[IP]
    else:
        q = qR
        Bstar = qL[IBX]
        # pout = qL[IP]

    beta_min = 1.0e-3
    alfven_max = 10.0
    beta = q[IP] / (0.5 * (q[IBX] * q[IBX] + q[IBY] * q[IBY] + q[IBZ] * q[IBZ]))
    alfven_number = np.sqrt(
        q[IR] * q[IU] / (q[IBX] * q[IBX] + q[IBY] * q[IBY] + q[IBZ] * q[IBZ])
    )
    is_low_beta = beta < beta_min
    u: State = primToCons(q)  # type: ignore
    # 3. Commpute flux
    flux = State()
    uS = Ustar[IDir.IX]
    flux[IR] = u[IR] * uS
    flux[IU] = u[IU] * uS + Pstar[IDir.IX]
    flux[IV] = u[IV] * uS + Pstar[IDir.IY]
    flux[IW] = u[IW] * uS + Pstar[IDir.IZ]
    flux[IE] = (
        u[IE] * uS
        + Pstar[IDir.IX] * uS
        + Pstar[IDir.IY] * Ustar[IDir.IY]
        + Pstar[IDir.IZ] * Ustar[IDir.IZ]
    )
    if is_low_beta or (alfven_number > alfven_max):
        flux[IBX] = u[IBX] * uS - q[IBX] * Ustar[IDir.IX]
        flux[IBY] = u[IBY] * uS - q[IBX] * Ustar[IDir.IY]
        flux[IBZ] = u[IBZ] * uS - q[IBX] * Ustar[IDir.IZ]
    else:
        flux[IBX] = u[IBX] * uS - Bstar * Ustar[IDir.IX]
        flux[IBY] = u[IBY] * uS - Bstar * Ustar[IDir.IY]
        flux[IBZ] = u[IBZ] * uS - Bstar * Ustar[IDir.IZ]
    # flux[IPSI] = 0.0
    # pout = Pstar[IDir.IX]
    return flux


# Calling the right Riemann solver
def riemann(qL: State, qR: State) -> State:
    match (params.riemann_solver.upper()):
        case "HLL":
            assert params.MHD is False, "HLL is not suitable for solving MHD problem."
            flux = hll(qL, qR)
        case "FIVEWAVES":
            flux = fivewaves(qL, qR)
        case _:
            raise ValueError("The selected Riemann solver is not available.")
    # case "HLLC": hllc(qL, qR, flux, pout, params); break;
    return flux
