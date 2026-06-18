#!/usr/bin/env python3
r"""
lsearch.py -- searching for L-functions from their functional equation and Euler
product, using the pole-free `coefficient_relation` of afe.py to generate the
defining equations.

This is the foundation (milestones M0 and M1) of the search tool:

  M0  data model: a `Landscape` (a functional equation whose conductor is known and
      whose Gamma factors have unknown spectral parameters), an `EulerProduct`
      (the shape that says which Dirichlet coefficients are independent unknowns),
      a list of weight functions, and a ground-truth `KnownTarget` oracle.

  M1  the Euler-product coefficient algebra: the map from the independent unknowns
      (one a_p per prime) to the full Dirichlet coefficient vector b(1..M), and
      back, together with the bookkeeping (accuracy -> number of terms M -> number
      of unknowns).

The first landscape is the 2-dimensional family of degree-3, conductor-1 L-functions
with Gamma factors

      Gamma_R(s + i*lambda1) Gamma_R(s + i*lambda2) Gamma_R(s - i*(lambda1+lambda2)).

Its ground truth is the GL(3) Maass form already stored in afe.py (GL3_MAASS), whose
spectral parameters are (lambda1, lambda2) = (-16.40312474..., -0.17112189...).

Conventions fixed for the whole search (per the project design):
  * All unknowns and equations are real: a_p -> (Re a_p, Im a_p); a complex
    equation -> its (Re, Im) pair.
  * Two real unknowns per prime.  For a tempered degree-3 form with trivial
    central character the Satake roots lie on the unit circle with product 1, so
    the local factor is  1 - a_p X + conj(a_p) X^2 - X^3  and every higher
    coefficient is determined by a_p:
        b(p^k) = a_p b(p^{k-1}) - conj(a_p) b(p^{k-2}) + b(p^{k-3}),   b(p^0)=1.
  * The sign epsilon is itself unknown, carried as (epsR, epsI) with the extra
    equation epsR^2 + epsI^2 = 1.
  * Equations come from the list of weight functions (not from varying a point s);
    `coefficient_relation` is always called at the fixed symmetry point s = 1/2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import time

import mpmath
from mpmath import mp, mpf, mpc

from afe import (coefficient_relation, coefficient_relation_grid,
                 GL3_MAASS, gl3_maass_bcoeffs)

# The fixed evaluation point for every coefficient_relation call (symmetry point).
FIXED_S = mpf(1) / 2


# ---------------------------------------------------------------------------
# Prime helpers
# ---------------------------------------------------------------------------

def primes_up_to(M):
    """List of primes <= M."""
    M = int(M)
    if M < 2:
        return []
    sieve = bytearray([1]) * (M + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(M ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, M + 1) if sieve[i]]


def prime_pi(M):
    """Number of primes <= M."""
    return len(primes_up_to(M))


def _smallest_prime_factor_table(M):
    spf = list(range(M + 1))
    i = 2
    while i * i <= M:
        if spf[i] == i:
            for j in range(i * i, M + 1, i):
                if spf[j] == j:
                    spf[j] = i
        i += 1
    return spf


# ---------------------------------------------------------------------------
# M0.  Data model
# ---------------------------------------------------------------------------

@dataclass
class Landscape:
    """A functional-equation family: known conductor, Gamma factors with unknown
    spectral parameters.  A 'point' is the tuple of spectral parameters."""
    name: str
    degree: int
    conductor: int
    dim: int                         # number of free spectral parameters
    mu_from_point: Callable          # point -> list of Gamma_R shifts (mu)
    nu_from_point: Callable          # point -> list of Gamma_C shifts (nu)


@dataclass
class EulerProduct:
    """The Euler-product shape: how the full coefficient vector is built from the
    independent per-prime unknowns, and how to read those unknowns back out."""
    name: str
    degree: int
    real_unknowns_per_prime: int
    bcoeffs_from_ap: Callable        # (ap: dict {p: a_p}, M) -> [b(1), ..., b(M)]
    extract_ap: Callable             # ([b(1), ..., b(M)]) -> {p: a_p}


@dataclass
class KnownTarget:
    """A fully known L-function sitting in a landscape, used as ground truth."""
    name: str
    landscape: Landscape
    euler: EulerProduct
    point: tuple                     # the true spectral parameters
    epsilon: mpc                     # the true sign
    ap: dict                         # the true {p: a_p}


# ---------------------------------------------------------------------------
# M1.  Euler-product coefficient algebra for the degree-3 tempered family
# ---------------------------------------------------------------------------

def gl3_bppow(a, k):
    """b(p^k) for the tempered degree-3 local factor 1 - a X + conj(a) X^2 - X^3.

    b(p^k) = a b(p^{k-1}) - conj(a) b(p^{k-2}) + b(p^{k-3}),  b(p^0)=1."""
    a = mpc(a)
    ac = mpmath.conj(a)
    h = [mpc(1)]
    for j in range(1, int(k) + 1):
        hj = a * h[j - 1]
        if j >= 2:
            hj -= ac * h[j - 2]
        if j >= 3:
            hj += h[j - 3]
        h.append(hj)
    return h[int(k)]


def gl3_bcoeffs_from_ap(ap, M):
    """Full coefficient vector b(1..M) from the per-prime unknowns ap = {p: a_p}.

    b(1)=1; b(p^k) by the degree-3 recurrence; composites by multiplicativity."""
    M = int(M)
    spf = _smallest_prime_factor_table(M)
    b = [mpc(0)] * (M + 1)
    if M >= 1:
        b[1] = mpc(1)
    for n in range(2, M + 1):
        m, val = n, mpc(1)
        while m > 1:
            p = spf[m]
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            # a_p defaults to 0 for primes not among the unknowns (their coefficients
            # are below the noise level), which drops those n from the series.
            val *= gl3_bppow(ap.get(p, mpc(0)), e)
        b[n] = val
    return b[1:M + 1]


def gl3_extract_ap(b):
    """Read the per-prime unknowns a_p = b(p) out of a coefficient vector b(1..M)."""
    return {p: mpc(b[p - 1]) for p in primes_up_to(len(b))}


GL3_LANDSCAPE = Landscape(
    name="GL(3,Z) Maass forms, conductor 1",
    degree=3, conductor=1, dim=2,
    mu_from_point=lambda pt: [1j * mpf(pt[0]), 1j * mpf(pt[1]),
                              -1j * (mpf(pt[0]) + mpf(pt[1]))],
    nu_from_point=lambda pt: [],
)

GL3_EULER = EulerProduct(
    name="degree 3, tempered, trivial central character",
    degree=3, real_unknowns_per_prime=2,
    bcoeffs_from_ap=gl3_bcoeffs_from_ap, extract_ap=gl3_extract_ap,
)


def gl3_known_target():
    """Ground truth: the first SL(3,Z) Maass form (LMFDB 3-1-1.1-r0e3-...), as a
    point of GL3_LANDSCAPE with its true sign and a_p."""
    # spectral parameters: GL3_MAASS["mu"] = [i*l1, i*l2, i*l3] with l3 = -(l1+l2)
    lam = [mpmath.re(-1j * m) for m in GL3_MAASS["mu"]]
    point = (lam[0], lam[1])
    b = gl3_maass_bcoeffs(40)        # n<=40 uses primes up to 37 (all stored)
    ap = gl3_extract_ap(b)
    return KnownTarget(name="SL(3,Z) Maass form 3-1-1.1-r0e3-...",
                       landscape=GL3_LANDSCAPE, euler=GL3_EULER,
                       point=point, epsilon=mpc(1), ap=ap)


# ---------------------------------------------------------------------------
# Weight functions
# ---------------------------------------------------------------------------

@dataclass
class WeightSet:
    """A list of weight functions g(s) = s^m exp(i*beta*s + alpha*s^2), as the
    parameter triples (m, alpha, beta).  Re(alpha) > 0 keeps every g admissible
    (Gaussian decay), and varying m and beta produces independent equations."""
    weights: list          # list of (m, alpha, beta)

    def __len__(self):
        return len(self.weights)


def admissibility_bound(landscape):
    """The phase bound (pi/2) * sum(kappa): a weight g=s^m exp(i*beta*s) (alpha=0)
    is admissible for |beta| < this.  kappa = 1/2 per Gamma_R, 1 per Gamma_C."""
    zero = (mpf(0),) * landscape.dim
    n_R = len(landscape.mu_from_point(zero))
    n_C = len(landscape.nu_from_point(zero))
    return mpmath.pi / 2 * (mpf(n_R) / 2 + n_C)


def _beta_values(beta_max):
    """Phases from beta_max down to 0 (and negatives), spaced FINER near beta_max
    (close, large beta still give distinct equations; small beta that are close give
    nearly identical ones)."""
    frac = [mpf(f) for f in ("1.0", "0.93", "0.86", "0.78", "0.68", "0.55", "0.38", "0.18")]
    vals = [beta_max * f for f in frac] + [mpf(0)]
    out = []
    for v in vals:
        out.append(v)
        if v != 0:
            out.append(-v)
    return out


def default_weight_set(n_weights, beta_max=mpf("1.5"), alphas=(mpf(0), mpf("0.02")),
                       m_max=5):
    """A DIVERSE pool of weights g(s)=s^m exp(i*beta*s + alpha*s^2) spanning a grid in
    (m, alpha, beta) -- the diversity is what makes the equations independent enough to
    resolve many Euler coefficients.  The solve then picks a well-conditioned subset
    (see solve_at_point), so the pool just needs to be diverse, not ordered.

    alpha in {0, <=1/50}: Re(alpha)>0 slows the n-decay and would inflate the term count
    M, but at the admissibility-limited beta the most demanding weight already fixes M,
    so a small alpha is essentially free here.  beta is kept >= ~1 below the
    admissibility bound (nearer the bound the integrand decays slowly and M explodes);
    larger |beta| are spaced finer (see _beta_values)."""
    betas = _beta_values(beta_max)
    alphas = [mpc(a) for a in alphas]
    grid = [(m, al, b) for m in range(m_max + 1) for al in alphas for b in betas]
    if n_weights >= len(grid):
        return WeightSet(weights=grid)
    # even stride through the grid -> a sample diverse in m, alpha and beta
    step = mpf(len(grid)) / n_weights
    idx = sorted(set(int(i * step) for i in range(n_weights)))
    return WeightSet(weights=[grid[i] for i in idx])


# ---------------------------------------------------------------------------
# Sizing: number of Dirichlet terms and number of unknowns
# ---------------------------------------------------------------------------

def dirichlet_length(landscape, point, weights, accuracy, working_precision=None,
                     epsilon=1):
    """Number of Dirichlet terms M needed at `point` for the given accuracy: the
    maximum truncation length over the weight list (so every equation is resolved
    at the common M)."""
    mu = landscape.mu_from_point(point)
    nu = landscape.nu_from_point(point)
    M = 0
    for (m, al, be) in weights.weights:
        r = coefficient_relation(FIXED_S, mu=mu, nu=nu, N=landscape.conductor,
                                 epsilon=epsilon, poles=[], g_m=m, g_alpha=al, g_beta=be,
                                 accuracy=accuracy, working_precision=working_precision)
        M = max(M, r.num_terms)
    return M


def count_unknowns(euler, M):
    """Number of real unknowns: 2 (for epsilon) + (real per prime) * pi(M)."""
    return 2 + euler.real_unknowns_per_prime * prime_pi(M)


# ---------------------------------------------------------------------------
# M2.  Equation assembly: the real residual system R(unknowns)
# ---------------------------------------------------------------------------
#
# Unknown real vector:   u = [epsR, epsI, x_p2, y_p2, x_p3, y_p3, ...]
# with epsilon = epsR + i epsI and a_p = x_p + i y_p for each prime p <= M.
# (b(1)=1 and the Euler product fix every other coefficient -- see M1.)
#
# Each weight g gives one COMPLEX equation.  Calling coefficient_relation with
# epsilon=1 yields A_n (coeff of b(n)) and B_n (coeff of conj(b(n)), epsilon-free
# base), so for a general epsilon the equation is
#
#     E_g(u) = sum_n A_n b(n)  +  epsilon * sum_n B_n conj(b(n))  +  pole_term  = 0,
#
# linear in epsilon.  Its (Re, Im) parts are two real equations.  Finally the sign
# normalization  epsR^2 + epsI^2 - 1 = 0  is appended.  Each equation is rescaled
# so its leading coefficients are O(1) (conditioning).

@dataclass
class EquationSystem:
    landscape: Landscape
    euler: EulerProduct
    point: tuple
    accuracy: int
    working_precision: int
    M: int                       # Dirichlet length
    primes: list                 # primes <= M  (the a_p unknowns, in order)
    n_unknowns: int              # U = 2 + 2*len(primes)
    A: list                      # A[i] = [A_1..A_M] for weight i (rescaled)
    B: list                      # B[i] = [B_1..B_M] for weight i (rescaled)
    pole: list                   # pole[i] (rescaled)
    scales: list                 # the rescaling factor used for weight i
    weights: list                # the (m, alpha, beta) actually used
    solve_idx: list              # indices into the residual vector used to solve
    detector_idx: list           # indices used as detectors
    sign: Optional[mpc] = None   # if not None, pin epsilon to this value


def pack_unknowns(epsilon, ap, primes):
    """(epsilon, {p: a_p}) -> real vector [epsR, epsI, x_p, y_p, ...]."""
    u = [mpmath.re(epsilon), mpmath.im(epsilon)]
    for p in primes:
        a = mpc(ap[p])
        u += [mpmath.re(a), mpmath.im(a)]
    return u


def unpack_unknowns(u, primes):
    """Inverse of pack_unknowns: -> (epsilon, {p: a_p})."""
    epsilon = mpc(u[0], u[1])
    ap = {p: mpc(u[2 + 2 * k], u[2 + 2 * k + 1]) for k, p in enumerate(primes)}
    return epsilon, ap


def build_equation_system(landscape, euler, point, accuracy, working_precision=None,
                          poles=(), sign=None, n_detectors=8, lead_terms=5,
                          weight_set=None):
    """Precompute the residual system at `point` (the expensive Gamma/g work).

    If `weight_set` is given it is used as-is; otherwise a default set sized to the
    number of unknowns is generated."""
    mu = landscape.mu_from_point(point)
    nu = landscape.nu_from_point(point)
    bmax = admissibility_bound(landscape) - mpf("1.0")   # keep beta well clear of the bound
    #   (right at bound-0.8 the integrand converges very slowly: M jumps from ~27 to ~160)

    # one auto-truncating call (most demanding weight) fixes the Dirichlet length M
    Msz = coefficient_relation(FIXED_S, mu=mu, nu=nu, N=landscape.conductor, epsilon=1,
                               poles=list(poles), g_m=0, g_alpha=0, g_beta=bmax,
                               accuracy=accuracy, working_precision=working_precision)
    M = Msz.num_terms

    if weight_set is not None:
        wlist = list(weight_set.weights)
    else:
        # Generate a POOL larger than the solve+detector equations need, so the solve
        # can pick a well-conditioned subset (QR row selection) and reach more primes.
        need_eqs = count_unknowns(euler, M) + n_detectors
        nW = max(28, -(-need_eqs // 2) + 8)        # ceil(need/2) + selection headroom
        wlist = list(default_weight_set(nW, beta_max=bmax).weights)
    weights = WeightSet(weights=wlist)

    # ONE shared Gamma grid for all weights (the big speed-up)
    rows = coefficient_relation_grid(FIXED_S, mu, nu, landscape.conductor, 1,
                                     list(poles), wlist, M, working_precision)

    A, B, pole, scales = [], [], [], []
    for (Ar, Br, pr) in rows:
        An = [mpc(x) for x in Ar]
        Bn = [mpc(x) for x in Br]
        k = min(lead_terms, M)
        lead = mpmath.fsum(abs(An[j]) + abs(Bn[j]) for j in range(k)) / (2 * k)
        sc = (1 / lead) if lead > 0 else mpf(1)
        A.append([x * sc for x in An])
        B.append([x * sc for x in Bn])
        pole.append(mpc(pr) * sc)
        scales.append(sc)

    # Select the determinable unknown primes: a_p enters the equations through the
    # coefficients A_p (of b(p)) and B_p (of conj b(p)); if these are below the
    # noise level ~10^{-accuracy} for every weight, a_p is not constrained -- its
    # Jacobian column is ~0 and the system would be singular.  Keep only the primes
    # above that level; the rest contribute < 10^{-accuracy} and are set to 0.
    unknown_tol = mpf(10) ** (-accuracy)
    sig = {p: max(max(abs(A[i][p - 1]), abs(B[i][p - 1])) for i in range(len(A)))
           for p in primes_up_to(M)}
    primes = [p for p in primes_up_to(M) if sig[p] > unknown_tol]
    U = 2 + euler.real_unknowns_per_prime * len(primes)

    # residual layout: [Re E_0, Im E_0, ..., Re E_{K-1}, Im E_{K-1}, normalization]
    K = len(weights.weights)
    norm_idx = 2 * K
    detector_idx = list(range(2 * K - n_detectors, 2 * K))      # last 8 coeff reals
    solve_pool = [i for i in range(2 * K) if i not in detector_idx]
    solve_idx = solve_pool[:U - 1] + [norm_idx]                 # U solve equations
    return EquationSystem(landscape=landscape, euler=euler, point=point,
                          accuracy=accuracy, working_precision=working_precision or 0,
                          M=M, primes=primes, n_unknowns=U, A=A, B=B, pole=pole,
                          scales=scales, weights=weights.weights,
                          solve_idx=solve_idx, detector_idx=detector_idx, sign=sign)


def residual(system, u, primes=None):
    """Full real residual vector at unknowns u (length 2*n_weights + 1, plus a sign
    pin if system.sign is set).  `primes` selects which primes u parametrizes
    (default system.primes); primes not in the list have a_p = 0."""
    if primes is None:
        primes = system.primes
    epsilon, ap = unpack_unknowns(u, primes)
    b = system.euler.bcoeffs_from_ap(ap, system.M)
    bconj = [mpmath.conj(x) for x in b]
    out = []
    for i in range(len(system.A)):
        Ai, Bi = system.A[i], system.B[i]
        E = system.pole[i]
        E += mpmath.fsum(Ai[n] * b[n] for n in range(system.M))
        E += epsilon * mpmath.fsum(Bi[n] * bconj[n] for n in range(system.M))
        out.append(mpmath.re(E))
        out.append(mpmath.im(E))
    out.append(u[0] * u[0] + u[1] * u[1] - 1)              # |epsilon|^2 = 1
    if system.sign is not None:                            # optional: pin the sign
        out.append(u[0] - mpmath.re(system.sign))
        out.append(u[1] - mpmath.im(system.sign))
    return out


def jacobian(system, u, h=None):
    """Numerical Jacobian of `residual` at u (central differences)."""
    if h is None:
        h = mpf(10) ** (-(system.working_precision // 2 or 12))
    r0 = residual(system, u)
    J = [[mpf(0)] * len(u) for _ in range(len(r0))]
    for j in range(len(u)):
        up = list(u); up[j] += h
        um = list(u); um[j] -= h
        rp = residual(system, up)
        rm = residual(system, um)
        for i in range(len(r0)):
            J[i][j] = (rp[i] - rm[i]) / (2 * h)
    return J


def _condition_number(rows):
    """Condition number of a real matrix (list of rows) via eigenvalues of A^T A."""
    A = mpmath.matrix(rows)
    AtA = A.T * A
    ev = mpmath.eigsy(AtA, eigvals_only=True)
    lo = min(abs(x) for x in ev)
    hi = max(abs(x) for x in ev)
    return mpmath.sqrt(hi / lo) if lo > 0 else mpmath.inf


def _select_rows(rows, k):
    """Greedy QR-style row pivoting: pick k rows that span the column space best.  Each
    step adds the row whose component orthogonal to the rows already chosen is largest;
    this maximizes the smallest singular value of the chosen submatrix, i.e. picks the
    most independent (best-conditioned) equations from a redundant pool."""
    k = min(k, len(rows))
    chosen, basis, avail = [], [], list(range(len(rows)))
    for _ in range(k):
        best, best_norm, best_v = -1, mpf(-1), None
        for i in avail:
            v = rows[i][:]
            for b in basis:                      # project out the chosen directions
                d = mpmath.fsum(v[t] * b[t] for t in range(len(v)))
                v = [v[t] - d * b[t] for t in range(len(v))]
            nv = mpmath.sqrt(mpmath.fsum(x * x for x in v))
            if nv > best_norm:
                best, best_norm, best_v = i, nv, v
        if best < 0 or best_norm == 0:
            break
        chosen.append(best)
        avail.remove(best)
        basis.append([x / best_norm for x in best_v])
    return chosen


def _equation_jacobian(system, x, primes, n_eq, deadline=None):
    """Numerical Jacobian (central differences) of the first n_eq residual rows with
    respect to the unknowns x.  Used to choose a well-conditioned solve/detector subset.
    Stops early (leaving later columns zero) if the wall-clock `deadline` is passed --
    the caller then sees the deadline and bails."""
    h = mpf(10) ** (-(system.working_precision // 2 or 12))
    U = len(x)
    rows = [[mpf(0)] * U for _ in range(n_eq)]
    for j in range(U):
        if deadline is not None and time.time() > deadline:
            break
        xp = x[:]; xp[j] += h
        xm = x[:]; xm[j] -= h
        rp = residual(system, xp, primes=primes)
        rm = residual(system, xm, primes=primes)
        for i in range(n_eq):
            rows[i][j] = (rp[i] - rm[i]) / (2 * h)
    return rows


def _select_from_jacobian(jac_full, K, euler, k, n_detectors):
    """Choose a well-conditioned square solve set (U-1 equations + the sign
    normalization) and n_detectors further independent equations for k unknown primes,
    by QR row pivoting on the columns of the full equation Jacobian belonging to those
    primes.  Returns (solve_idx, det_idx)."""
    U = 2 + euler.real_unknowns_per_prime * k
    cols = [0, 1] + [2 + 2 * j + t for j in range(k) for t in (0, 1)]   # eps + k primes
    sub = [[row[c] for c in cols] for row in jac_full]
    sel = _select_rows(sub, (U - 1) + n_detectors)
    solve_idx = sel[:U - 1] + [2 * K]                      # + the sign normalization
    det_idx = sel[U - 1:U - 1 + n_detectors]
    return solve_idx, det_idx


# ---------------------------------------------------------------------------
# M3.  Solver: a secant (Broyden) iteration on a restricted unknown set
# ---------------------------------------------------------------------------
#
# The unknowns are restricted to the determinable primes (above the rank gap); if
# the solve does not reach the target residual the set is enlarged one prime at a
# time.  Broyden ("secant", not Newton): one finite-difference Jacobian at the
# start, then rank-1 secant updates -- so the Jacobian is not re-differenced every
# step and little precision is lost.

def _guess_vector(primes, guess, eps_guess):
    ap = {p: mpc((guess or {}).get(p, 0)) for p in primes}
    return pack_unknowns(eps_guess, ap, primes)


def _broyden(F, x0, h, tol, maxiter, deadline=None, stall_window=40):
    """Solve F(x)=0 by Broyden's (good) method.  Returns (x, ||F||) of the best
    iterate found (caller decides if it is good enough); None if B goes singular.

    Two early exits keep a non-converging solve (e.g. far from any L-function) from
    burning the full maxiter: it stops if the best residual has not improved for
    stall_window iterations (a genuine convergence keeps improving, so this is
    false-negative-safe), or if the wall-clock `deadline` (epoch seconds) is passed."""
    n = len(x0)
    x = [mpf(v) for v in x0]
    Fx = F(x)
    nrm = max(abs(v) for v in Fx)
    best = (x[:], nrm)
    if nrm < tol:
        return best
    B = mpmath.matrix(n, n)                     # initial Jacobian, forward differences
    for j in range(n):
        if deadline is not None and time.time() > deadline:
            return best
        xp = x[:]
        xp[j] = xp[j] + h
        Fp = F(xp)
        for i in range(n):
            B[i, j] = (Fp[i] - Fx[i]) / h
    Fxv = mpmath.matrix(Fx)
    last_improve = 0
    for it in range(maxiter):
        if deadline is not None and time.time() > deadline:
            return best
        try:
            dx = mpmath.lu_solve(B, -Fxv)
        except (ZeroDivisionError, ValueError):
            return None
        x = [x[i] + dx[i] for i in range(n)]
        Fn = F(x)
        nrm = max(abs(v) for v in Fn)
        if nrm < best[1]:
            best = (x[:], nrm)
            last_improve = it
        if nrm < tol:
            return (x[:], nrm)
        if it - last_improve > stall_window:   # plateaued, not converging -> give up
            return best
        Fnv = mpmath.matrix(Fn)
        dF = Fnv - Fxv
        denom = mpmath.fsum(dx[i] * dx[i] for i in range(n))
        if denom == 0:
            break
        Bdx = B * dx
        corr = dF - Bdx
        for i in range(n):
            ci = corr[i]
            for j in range(n):
                B[i, j] = B[i, j] + ci * dx[j] / denom
        Fxv = Fnv
    return best


def solve_at_point(system, guess=None, eps_guess=None, k_init=3, k_max=None,
                   tol=None, det_tol=None, maxiter=150, n_detectors=8,
                   fixed=None, deadline=None, verbose=False):
    """Find a solution at system.point, restricting the unknown primes and enlarging
    the set if no solution is found.

    For each prime count the solve equations are chosen by QR row pivoting from the
    (redundant) weight pool, so a well-conditioned square subset is used instead of an
    arbitrary one -- this is what lets the solve reach many primes and drive the
    detector residual to the noise floor.  Pass `fixed=(solve_idx, det_idx, primes)` to
    reuse a previously chosen equation set (e.g. the same detectors at every box corner).

    Returns a dict (primes, epsilon, ap, residuals, solve_idx, det_idx, ...) or None."""
    cand = system.primes
    if not cand:
        return None
    if eps_guess is None:
        eps_guess = mpc(1)
    if tol is None:                 # the square solve set must be driven near 0
        tol = mpf(10) ** (-(system.accuracy))
    if det_tol is None:             # a genuine solution drives the detectors to ~floor
        det_tol = mpf(10) ** (-(system.accuracy - 1))
    cond_cap = mpf(10) ** (system.working_precision / 2)   # don't lose > half the digits
    h = mpf(10) ** (-(system.working_precision // 3))

    K = len(system.A)
    if fixed is not None:           # reuse a chosen equation set at a fixed prime count
        solve_idx, det_idx, primes = fixed
        ks = [len(primes)]
    else:
        if k_max is None:
            k_max = len(cand)
        k_init = max(1, min(k_init, len(cand)))
        ks = list(range(k_init, k_max + 1))
        # one equation Jacobian over ALL candidate primes; per-k selection reuses its
        # columns (far cheaper than re-differencing for every prime count)
        xall = _guess_vector(cand, guess, eps_guess)
        jac_full = _equation_jacobian(system, xall, cand, 2 * K, deadline=deadline)

    best = None
    for k in ks:
        if deadline is not None and time.time() > deadline:
            break                       # out of time -> return the best found so far
        if fixed is None:
            primes = cand[:k]
            solve_idx, det_idx = _select_from_jacobian(jac_full, K, system.euler,
                                                       k, n_detectors)
        x0 = _guess_vector(primes, guess, eps_guess)

        def F(u, _si=solve_idx, _pr=primes):
            full = residual(system, u, primes=_pr)
            return [full[i] for i in _si]

        out = _broyden(F, x0, h, tol, maxiter, deadline=deadline)
        if out is None:             # Jacobian went singular -> set too large, stop growing
            if verbose:
                print(f"   k={k} primes={primes}: Broyden singular")
            break
        x, nrm = out
        if nrm > tol:               # solve set didn't converge -> enlarge
            if verbose:
                print(f"   k={k} primes={primes}: solve stalled {mpmath.nstr(nrm,3)}")
            continue
        cond = _solve_cond(system, primes, x, solve_idx, h)
        if cond > cond_cap and best is not None:
            if verbose:                # this k is rank-deficient -> stop enlarging
                print(f"   k={k} primes={primes}: cond {mpmath.nstr(cond,2)} > cap, stop")
            break
        full = residual(system, x, primes=primes)
        det = max(abs(full[i]) for i in det_idx) if det_idx else mpf(0)
        if verbose:
            print(f"   k={k} primes={primes}: ||solve||={mpmath.nstr(nrm,3)}  "
                  f"||det||={mpmath.nstr(det,3)}  cond={mpmath.nstr(cond,2)}")
        eps, ap = unpack_unknowns(x, primes)
        cand_sol = {"primes": primes, "epsilon": eps, "ap": ap, "cond": cond,
                    "solve_res": nrm, "det_res": det, "x": x, "k": k,
                    "solve_idx": solve_idx, "det_idx": det_idx}
        if best is None or det < best["det_res"]:
            best = cand_sol
        if det < det_tol:           # detectors at the floor -> a genuine solution
            break
    return best


def solve_at_point_lsq(system, guess=None, eps_guess=None, k=None, n_eq=None,
                       tol=None, maxiter=40, damping=mpf("1e-30"), verbose=False):
    """PROTOTYPE overdetermined least-squares solve at system.point.

    Instead of a square system (U unknowns, U equations) it fits MANY more equations
    than unknowns by Gauss-Newton, minimizing the L2 residual over a QR-selected,
    well-conditioned set of equations (plus the sign normalization).  The redundancy
    averages down the per-equation noise, and the weakly-determined high-prime unknowns
    absorb their Dirichlet terms instead of biasing the well-determined low ones -- so
    the low coefficients and the spectral parameters come out sharper.

    Each Gauss-Newton step solves  min || J du + F ||  via QR (no normal-equation
    squaring); a tiny Levenberg-Marquardt damping stabilizes the near-dependent
    high-prime directions without perturbing the determined ones.  Returns a dict with
    the coefficients, the L2 residual of the fitted set, and the max residual over ALL
    equations (a global goodness-of-fit)."""
    cand = system.primes
    if not cand:
        return None
    if k is None:
        k = len(cand)
    primes = cand[:k]
    U = 2 + system.euler.real_unknowns_per_prime * k
    if eps_guess is None:
        eps_guess = mpc(1)
    if tol is None:
        tol = mpf(10) ** (-(system.accuracy))
    K = len(system.A)
    norm_idx = 2 * K
    # fit as many well-conditioned equations as asked (default: a generous multiple of
    # the unknowns), chosen by QR pivoting; always include the sign normalization
    if n_eq is None:
        n_eq = min(2 * K, 4 * U)
    x0 = _guess_vector(primes, guess, eps_guess)
    jac0 = _equation_jacobian(system, x0, primes, 2 * K)
    eq_idx = _select_rows(jac0, n_eq) + [norm_idx]
    x, nrm = _lsq_core(system, primes, eq_idx, x0, tol, maxiter, damping)

    eps, ap = unpack_unknowns(x, primes)
    full = residual(system, x, primes=primes)
    all_res = max(abs(full[i]) for i in range(2 * K))   # fit over EVERY equation
    if verbose:
        print(f"   lsq k={k}: n_eq={len(eq_idx)} ||fit||={mpmath.nstr(nrm,3)} "
              f"all_res={mpmath.nstr(all_res,3)}")
    return {"primes": primes, "epsilon": eps, "ap": ap, "x": x, "k": k,
            "lsq_res": nrm, "all_res": all_res, "n_eq": len(eq_idx)}


def _lsq_core(system, primes, fit_idx, x0, tol, maxiter, damping=mpf("1e-30")):
    """Gauss-Newton least-squares fit of the residual equations indexed by fit_idx,
    starting from x0.  The equations are nearly linear in the coefficients, so the
    Jacobian is computed ONCE at x0 and reused (chord / modified Gauss-Newton): each
    step solves min ||J du + F|| via QR (LM-damped normal equations as a rank-deficient
    fallback).  Returns (x, ||fit residual||)."""
    U = len(x0)
    h = mpf(10) ** (-(system.working_precision // 2))
    x = list(x0)

    def Fvec(u):
        full = residual(system, u, primes=primes)
        return [full[i] for i in fit_idx]

    # one Jacobian at x0, reused for every step
    J = [[mpf(0)] * U for _ in range(len(fit_idx))]
    for j in range(U):
        xp = x[:]; xp[j] += h
        xm = x[:]; xm[j] -= h
        rp = Fvec(xp); rm = Fvec(xm)
        for i in range(len(fit_idx)):
            J[i][j] = (rp[i] - rm[i]) / (2 * h)
    Jm = mpmath.matrix(J)

    def step(Fx):
        Fv = mpmath.matrix(Fx)
        try:                                    # least-squares step via QR
            return mpmath.qr_solve(Jm, -Fv)[0]
        except Exception:                       # rank-deficient -> damped normal eqs
            JtJ = Jm.T * Jm
            for i in range(U):
                JtJ[i, i] += damping
            return mpmath.lu_solve(JtJ, -(Jm.T * Fv))

    nrm = None
    for _ in range(maxiter):
        Fx = Fvec(x)
        nrm = max(abs(v) for v in Fx)
        if nrm < tol:
            break
        du = step(Fx)
        x = [x[i] + du[i] for i in range(U)]
    return x, nrm


# ---------------------------------------------------------------------------
# M4.  Box geometry: detectors -> hyperplanes -> candidate cloud
# ---------------------------------------------------------------------------
#
# A triangular box: one corner at the given point P, the others a box-size h away
# along each axis.  Solve at each corner (continuing from the corner-0 solution,
# keeping the same primes/weights/detectors).  Each real detector, as a function of
# position, vanishes at a true L-function; from its 3 corner values fit it affinely
# and take the line where it is 0.  Intersect the detector lines pairwise -> a cloud
# of candidate points, which should concentrate at the true (lambda1, lambda2).

def triangle_corners(point, h):
    """Right-triangle corners: P, P+(h,0), P+(0,h)."""
    h = mpf(h)
    p = (mpf(point[0]), mpf(point[1]))
    return [p, (p[0] + h, p[1]), (p[0], p[1] + h)]


def box_step(landscape, euler, point, boxsize, accuracy, working_precision,
             guess=None, eps_guess=None, solver="square", deadline=None, verbose=False):
    """One box iteration's geometry: returns the corner solutions, the detector
    lines, and the candidate cloud (Step 4-6).

    solver="square" (default) recovers the coefficients at each corner with the square
    QR-selected solve -- this is what the detector-line geometry wants, because the
    solve equations are driven exactly to zero so the held-out detectors' zero-lines
    pass cleanly through the true point.  solver="lsq" instead refines the corner
    coefficients by least-squares over all non-detector equations; this does NOT reliably
    improve the lambda-determination (the L2 compromise muddies the held-out detector
    geometry), so it is not the default.  LSQ's real value is sharper COEFFICIENTS at a
    known point -- see search(refine_coeffs=...)."""
    h = mpf(boxsize)
    corners = triangle_corners(point, h)

    timed_out = lambda: deadline is not None and time.time() > deadline

    sys0 = build_equation_system(landscape, euler, corners[0], accuracy, working_precision)
    sol0 = solve_at_point(sys0, guess=guess, eps_guess=eps_guess, deadline=deadline)
    if sol0 is None or timed_out():
        return None
    k = sol0["k"]
    weights = WeightSet(weights=sys0.weights)
    # the QR-chosen solve/detector equations at corner 0 are reused at every corner, so
    # the detector lines compare the SAME equations across the box
    detector_idx = sol0["det_idx"]
    fixed = (sol0["solve_idx"], sol0["det_idx"], sol0["primes"])

    systems, sols = [sys0], [sol0]
    for c in corners[1:]:                       # continue from corner-0 solution
        if timed_out():             # bail between corners so a timeout doesn't overshoot
            return None
        sysc = build_equation_system(landscape, euler, c, accuracy, working_precision,
                                     weight_set=weights)
        solc = solve_at_point(sysc, guess=sol0["ap"], eps_guess=sol0["epsilon"],
                              fixed=fixed, deadline=deadline)
        if solc is None:            # a corner solve failed -> cannot form the geometry
            return None
        systems.append(sysc)
        sols.append(solc)

    if solver == "lsq":
        # refine each corner's coefficients by least-squares over every non-detector
        # equation (plus the sign normalization), warm-started from the square solve.
        # The detector equations stay held out, so their zero-lines remain independent.
        primes = sol0["primes"]
        K = len(sys0.A)
        det_set = set(detector_idx)
        fit_idx = [i for i in range(2 * K) if i not in det_set] + [2 * K]
        tol = mpf(10) ** (-(sys0.accuracy))
        for i in range(3):
            x_ref, _ = _lsq_core(systems[i], primes, fit_idx, sols[i]["x"], tol, 30)
            eps_r, ap_r = unpack_unknowns(x_ref, primes)
            sols[i] = {**sols[i], "x": x_ref, "epsilon": eps_r, "ap": ap_r}

    # detector value vector at each corner (using that corner's own solution)
    Dvals = []
    for i in range(3):
        full = residual(systems[i], sols[i]["x"], primes=sols[i]["primes"])
        Dvals.append([full[d] for d in detector_idx])

    # affine model of each detector: D(d1,d2) = D0 + g1 d1 + g2 d2  (d = offset from P)
    lines = []
    for d in range(len(detector_idx)):
        D0, D1, D2 = Dvals[0][d], Dvals[1][d], Dvals[2][d]
        lines.append((D0, (D1 - D0) / h, (D2 - D0) / h))

    # pairwise intersection of the detector zero-lines -> cloud.  Record, per point,
    # the offset |Delta| from P and the inverse-matrix norm ||M^{-1}|| -- both needed
    # to estimate how much precision the intersection costs.
    cloud, cloud_info = [], []
    nd = len(lines)
    for i in range(nd):
        for j in range(i + 1, nd):
            D0i, g1i, g2i = lines[i]
            D0j, g1j, g2j = lines[j]
            det = g1i * g2j - g2i * g1j
            if abs(det) < mpf(10) ** (-(working_precision - 4)) * (abs(g1i) + abs(g2i) + 1):
                continue                        # near-parallel -> unstable, skip
            d1 = (-D0i * g2j + g2i * D0j) / det
            d2 = (-g1i * D0j + g1j * D0i) / det
            Minv_norm = max(abs(g2j) + abs(g2i), abs(g1j) + abs(g1i)) / abs(det)
            cloud.append((point[0] + d1, point[1] + d2))
            cloud_info.append({"offset": max(abs(d1), abs(d2)), "Minv": Minv_norm})

    # precision lost in recovering the coefficients (solve conditioning at corner 0)
    cond_solve = sol0["cond"]
    if verbose:
        print(f"   corner detector norms: "
              f"{[mpmath.nstr(max(abs(v) for v in Dvals[i]),2) for i in range(3)]}")
        print(f"   cloud size = {len(cloud)}, solve cond = {mpmath.nstr(cond_solve,3)}")
    return {"corners": corners, "sols": sols, "lines": lines, "cloud": cloud,
            "cloud_info": cloud_info, "detector_idx": detector_idx, "k": k,
            "h": h, "cond_solve": cond_solve, "working_precision": working_precision}


def _solve_cond(system, primes, x, solve_idx, h):
    """Condition number of the solve subsystem (d solve-equations / d unknowns) at x;
    log10 of it is roughly the number of digits lost in recovering the coefficients."""
    r0 = [residual(system, x, primes=primes)[i] for i in solve_idx]
    U = len(x)
    J = [[mpf(0)] * U for _ in range(len(solve_idx))]
    for j in range(U):
        xp = x[:]
        xp[j] = xp[j] + h
        rp = [residual(system, xp, primes=primes)[i] for i in solve_idx]
        for i in range(len(solve_idx)):
            J[i][j] = (rp[i] - r0[i]) / h
    return _condition_number(J)


def estimate_cloud_precision(box_result):
    """Estimated absolute numerical uncertainty of the cloud points (the precision
    LOST in the calculation), so the loop can raise the working precision.

    A cloud point p solves M p = -D0 (M = the two detector gradients, D0 the detector
    values).  Each detector value carries error ~ eps_D = 10^{-wp} * cond_solve (wp
    round-off amplified by the coefficient solve).  Propagating,
        |dp| ~ ||M^{-1}|| * eps_D * (1 + |offset|/h),
    the last factor being the gradient-fit cancellation when the box does not bracket
    the truth.  Returns the median such |dp| over the cloud."""
    wp = box_result["working_precision"]
    cond = box_result["cond_solve"]
    h = box_result["h"]
    eps_D = mpf(10) ** (-wp) * cond                 # detector value error (round-off x solve cond)
    amps = sorted(info["Minv"] * (1 + info["offset"] / h)
                  for info in box_result["cloud_info"])
    if not amps:
        return None
    # the robust cloud centre is set by the WELL-conditioned line pairs, not the
    # median (near-parallel pairs throw wild outliers the median ignores); use the
    # lower-quartile amplification.
    amp = amps[len(amps) // 4]
    return eps_D * amp


def cloud_center_spread(cloud, frac=mpf("0.67")):
    """Robust centre (coordinate-wise median) and spread = the radius (per axis) that
    contains a fraction `frac` of the points (Step 7: "a box containing MOST of the
    points") -- not the max, which near-parallel detector pairs make outlier-driven."""
    if not cloud:
        return None, None
    xs = sorted(p[0] for p in cloud)
    ys = sorted(p[1] for p in cloud)
    cx, cy = xs[len(xs) // 2], ys[len(ys) // 2]
    dx = sorted(abs(p[0] - cx) for p in cloud)
    dy = sorted(abs(p[1] - cy) for p in cloud)
    idx = min(len(cloud) - 1, int(frac * len(cloud)))
    spread = max(dx[idx], dy[idx])
    return (cx, cy), spread


# ---------------------------------------------------------------------------
# M5 + M6.  The iterative box search
# ---------------------------------------------------------------------------

def _average_solution(sols):
    """Average the corner solutions (Step 8: next guess = average over box points)."""
    sols = [s for s in sols if s is not None]
    primes = sols[0]["primes"]
    n = len(sols)
    ap = {p: mpmath.fsum(s["ap"][p] for s in sols) / n for p in primes}
    eps = mpmath.fsum(s["epsilon"] for s in sols) / n
    return {"primes": primes, "ap": ap, "epsilon": eps}


# Accuracy needed to resolve a box of half-width h: the detector lines must agree on
# the point a good bit finer than the box, or the lambda-cloud never shrinks below it.
# Empirically (GL(3), box 1e-3) accuracy = -log10(h) + ACC_OVER ~ 8 puts the cloud at
# ~box/3.  The accuracy is capped at what the *target* box needs (ACC_MARGIN digits past
# -log10(target_box) + ACC_OVER): asking for more is wasted work, and -- with the present
# weight set -- it also rank-collapses the solve (cond explodes; see lsearch-lambda2-floor).
ACC_OVER = mpf(8)
ACC_MARGIN = 2                  # accuracy headroom beyond what the target box needs
GUARD_DIGITS = 12               # wp must beat accuracy+log10(cond) by this many digits


def _acc_max_for_target(target_box, ln10):
    """Largest accuracy worth using to reach a box of half-width target_box."""
    return mpmath.ceil(-mpmath.log(target_box) / ln10 + ACC_OVER + ACC_MARGIN)


def _accuracy_for_box(boxsize, floor, acc_max, ln10):
    return min(mpf(acc_max), max(floor, -mpmath.log(boxsize) / ln10 + ACC_OVER))


def _report_iteration(it, center, spread, cloud_prec, accuracy, wp, boxsize,
                      cond, sol, det_res):
    """Per-iteration progress report while refining a point: the current spectral
    parameters and how well they are pinned down, the recovered Euler coefficients and
    the sign, and the accuracy/precision parameters driving the search."""
    # how many decimal places of lambda are actually determined (by the cloud spread)
    det_digits = int(max(0, -mpmath.log(spread) / mpmath.log(10))) if spread else 0
    show = det_digits + 6           # determined digits plus a few to watch them settle
    print(f"  ---- iteration {it} " + "-" * 40)
    # current best guess of the L-point coordinates and the box being searched
    print(f"  current guess: L-point = ({mpmath.nstr(center[0], show)}, "
          f"{mpmath.nstr(center[1], show)})   box size = {mpmath.nstr(boxsize, 2)}")
    print(f"  spectral parameters (determined to +- {mpmath.nstr(spread, 2)}):")
    for i, c in enumerate(center, 1):
        print(f"      lambda{i} = {mpmath.nstr(c, show)}")
    print(f"  spectral-parameter precision: determination +- {mpmath.nstr(spread, 2)}"
          f" ,  numerical {mpmath.nstr(cloud_prec, 2)}")
    print(f"  accuracy = {int(round(accuracy))} digits   working precision = {wp} digits"
          f"   box half-width = {mpmath.nstr(boxsize, 2)}")
    print(f"  sign epsilon = {mpmath.nstr(sol['epsilon'], show)}")
    print(f"  solve condition = {mpmath.nstr(cond, 3)}"
          f"   detector residual = {mpmath.nstr(det_res, 2)}"
          f"   primes used = {len(sol['primes'])}")
    print(f"  Euler coefficients a_p:")
    for p in sol["primes"]:
        print(f"      a_{p:<3d}= {mpmath.nstr(sol['ap'][p], 12)}")


def _finalize_coeffs(result, landscape, euler, refine_coeffs, verbose):
    """At the converged point, recover the Euler coefficients by least-squares (more
    equations than unknowns), which is where the LSQ solve genuinely helps: it averages
    out per-equation noise and lets the high primes absorb their terms, sharpening the
    low coefficients.  Replaces result['sol'] and records the global fit residual."""
    if not refine_coeffs:
        return result
    pt = result["point"]
    acc = result["accuracy"]
    wp = result["wp"]
    sol = result.get("sol") or {}
    mp.dps = wp
    sysf = build_equation_system(landscape, euler, pt, acc, wp)
    ls = solve_at_point_lsq(sysf, guess=sol.get("ap"), eps_guess=sol.get("epsilon"))
    if ls is not None:
        result["sol"] = {"primes": ls["primes"], "ap": ls["ap"], "epsilon": ls["epsilon"]}
        result["coeff_fit_res"] = ls["all_res"]
        if verbose:
            print(f"  [final coefficients by least-squares: {ls['n_eq']} equations, "
                  f"global fit residual {mpmath.nstr(ls['all_res'], 2)}]")
    return result


def search(landscape, euler, point, boxsize, accuracy, working_precision, target_box,
           guess=None, eps_guess=None, max_iter=40, wander_dist=mpf("0.25"),
           stall_factor=mpf("0.9"), stall_patience=2, solver="square",
           refine_coeffs=True, timeout=600, verbose=True):
    """Iterative search (Steps 1-8): shrink a triangular box toward an L-function,
    raising the accuracy (to tighten the spectral-parameter cloud) and the working
    precision (to keep the cloud points trustworthy) as the box shrinks.

    Stops with 'success' if the box reaches target_box, or 'converged' if the box stops
    shrinking -- the detector floor (prime truncation) has been hit and no more digits
    are available at this accuracy.  Returns a status dict.

    Example -- refine a point near the first SL(3,Z) Maass form in this landscape::

        import lsearch as L
        from mpmath import mpf, mpc

        t = L.gl3_known_target()                 # the GL(3) landscape + Euler product
        res = L.search(
            t.landscape, t.euler,
            point=(mpf('-16.4036'), mpf('-0.1716')),   # starting guess for (lambda1, lambda2)
            boxsize=mpf('1e-3'),                       # initial box half-width
            accuracy=8,                                # starting accuracy floor (digits)
            working_precision=30,                      # starting mpmath precision (digits)
            target_box=mpf('1e-6'),                    # stop once the box is this small
            guess=t.ap, eps_guess=mpc(1),              # initial Euler coeffs / sign guess
        )
        print(res['status'], res['point'])       # 'converged'/'success' and (lambda1, lambda2)

    A per-iteration progress report (spectral parameters and their precision, Euler
    coefficients, sign, accuracy, working precision, condition number, ...) is printed
    when ``verbose=True``.  For a cold search with no coefficient guess, pass
    ``guess=None`` (the unknowns start at 0)."""
    point = (mpf(point[0]), mpf(point[1]))
    boxsize = mpf(boxsize)
    target = mpf(target_box)
    acc_floor = mpf(accuracy)
    wp = int(working_precision)
    g, eg = guess, (eps_guess or mpc(1))
    start = point                                 # the grid point we started from
    wander_dist = mpf(wander_dist)
    ln10 = mpmath.log(10)
    acc_max = _acc_max_for_target(target, ln10)   # cap accuracy at what the target needs
    stalls = 0
    refining = False             # cheap exploration until the box first shrinks
    last = None                  # last accepted (center, new_box, wp, cloud_prec, sol)
    deadline = (time.time() + timeout) if timeout else None

    def _timed_out():
        # the time limit only guards the search for an INITIAL solution; once the box has
        # shrunk (refining), a genuine candidate is being polished -- never abandon it
        return deadline is not None and not refining and time.time() > deadline

    for it in range(max_iter):
        if _timed_out():         # ran out of wall-clock time without a solution
            return {"status": "timeout", "reason": "time limit", "iter": it,
                    **(last or {"point": point})}
        # Cheap exploration vs refinement.  Until the box has actually shrunk once (a
        # candidate is homing in) we EXPLORE at the accuracy floor: this keeps the
        # (atomic, uninterruptible) build cheap, so a far-from-anything point is judged
        # and abandoned in minutes instead of grinding for hours on big-M builds.  Only
        # after a shrink do we couple accuracy to the box and refine in earnest.
        if refining:
            accuracy = _accuracy_for_box(boxsize, acc_floor, acc_max, ln10)
        else:
            accuracy = acc_floor
        wp = max(wp, int(mpmath.ceil(accuracy)) + 6)

        # Box step with the precision guard.  The numerical error of every cloud point
        # and its genuine (accuracy-limited) spread are both amplified by the same
        # near-parallel factor, so the right guard is uniform:  wp >= accuracy +
        # log10(cond) + GUARD_DIGITS.  Redo at higher wp until it holds.
        for attempt in range(6):
            if _timed_out():         # don't start another (re)build past the deadline
                return {"status": "timeout", "reason": "time limit", "iter": it,
                        **(last or {"point": point})}
            mp.dps = wp
            bs = box_step(landscape, euler, point, boxsize, int(round(accuracy)), wp,
                          guess=g, eps_guess=eg, solver=solver,
                          deadline=(None if refining else deadline), verbose=False)
            if _timed_out():
                return {"status": "timeout", "reason": "time limit", "iter": it,
                        **(last or {"point": point})}
            if bs is None or not bs["cloud"]:
                return {"status": "fail", "reason": "no solution", "iter": it}
            cond = bs["cond_solve"]
            center, spread = cloud_center_spread(bs["cloud"])
            new_box = min(spread * mpf("1.2"), boxsize)
            need = int(mpmath.ceil(accuracy + mpmath.log(cond) / ln10 + GUARD_DIGITS))
            if wp >= need:
                break
            new_wp = need + 3       # 3 digits of headroom so we don't redo again at once
            if verbose:    # show the provisional guess so a redo isn't a silent wait
                print(f"   provisional guess: L-point=({mpmath.nstr(center[0], 12)}, "
                      f"{mpmath.nstr(center[1], 12)})  box size={mpmath.nstr(boxsize, 2)}")
                print(f"   (redo: wp {wp} < accuracy+log10(cond)+{GUARD_DIGITS} = {need}"
                      f" -> raising wp to {new_wp})")
            wp = new_wp
        cloud_prec = estimate_cloud_precision(bs)
        det_res = max((s["det_res"] for s in bs["sols"] if s), default=mpf(0))
        sol = _average_solution(bs["sols"])
        if verbose:
            _report_iteration(it, center, spread, cloud_prec, accuracy, wp,
                              boxsize, cond, sol, det_res)

        # Wander check: give up if the cloud centre has drifted more than an ABSOLUTE
        # distance wander_dist from the grid point we started at.  Absolute (not box-
        # relative) so a search can begin on a grid much coarser than the box and still
        # follow the detectors to a nearby L-function, abandoning only genuine runaways.
        if max(abs(center[0] - start[0]), abs(center[1] - start[1])) > wander_dist:
            return {"status": "fail", "reason": "wandered", "iter": it, "point": center,
                    "box": new_box}

        last = {"point": center, "box": new_box, "iter": it,
                "accuracy": int(round(accuracy)), "wp": wp,
                "cloud_prec": cloud_prec, "det_res": det_res, "sol": sol}

        if new_box <= target:
            return _finalize_coeffs({"status": "success", **last},
                                    landscape, euler, refine_coeffs, verbose)

        # Stall detection: the box stopped shrinking meaningfully -> the detector floor
        # has been reached; report the converged point at the achievable precision.
        # A meaningful shrink is also the signal that a real candidate is here, so we
        # switch from cheap exploration to full-accuracy refinement.
        if new_box > stall_factor * boxsize:
            stalls += 1
            if stalls >= stall_patience:
                return _finalize_coeffs(
                    {"status": "converged", "reason": "detector floor", **last},
                    landscape, euler, refine_coeffs, verbose)
        else:
            stalls = 0
            if not refining:
                refining = True
                if verbose:
                    print("   (box shrank -> switching from exploration to refinement)")

        if verbose and new_box < boxsize:
            print(f"   box size decreased: {mpmath.nstr(boxsize, 2)} -> "
                  f"{mpmath.nstr(new_box, 2)}")
        avg = last["sol"]
        g, eg = avg["ap"], avg["epsilon"]
        point, boxsize = center, new_box
    return {"status": "fail", "reason": "max_iter", "iter": max_iter,
            **(last or {"point": point})}


# ---------------------------------------------------------------------------
# Self-test for M0 + M1
# ---------------------------------------------------------------------------

def selftest(accuracy=8, working_precision=30, verbose=True):
    """Check the data model and the Euler algebra against the known GL(3) target."""
    results = []
    old = mp.dps
    mp.dps = working_precision
    try:
        target = gl3_known_target()
        land, eul = target.landscape, target.euler

        # --- M0: the landscape point reproduces the stored Gamma shifts ---------
        mu = land.mu_from_point(target.point)
        mu_err = max(abs(mu[i] - GL3_MAASS["mu"][i]) for i in range(3))
        results.append(("point -> mu reproduces GL3_MAASS", mu_err, mpf(10) ** (-14)))
        if verbose:
            print(f"[M0] point={tuple(mpmath.nstr(x,10) for x in target.point)}")
            print(f"     mu reconstruction error = {mpmath.nstr(mu_err,3)}")

        # --- M1: round-trip the coefficients through a_p ------------------------
        M0 = 40
        b_true = gl3_maass_bcoeffs(M0)
        ap = eul.extract_ap(b_true)
        b_recon = eul.bcoeffs_from_ap(ap, M0)
        rt_err = max(abs(b_recon[i] - b_true[i]) for i in range(M0))
        results.append(("a_p -> b(n) round-trip (n<=40)", rt_err, mpf(10) ** (-13)))
        if verbose:
            print(f"[M1] coefficient round-trip max error (n<=40) = {mpmath.nstr(rt_err,3)}")
            print(f"     a_2 = {mpmath.nstr(ap[2],8)}   b(4) = a_2^2-conj(a_2) ?"
                  f" {mpmath.nstr(b_recon[3],8)} vs {mpmath.nstr(ap[2]**2-mpmath.conj(ap[2]),8)}")

        # --- M1: sizing -------------------------------------------------------
        weights = default_weight_set(14)
        M = dirichlet_length(land, target.point, weights, accuracy, working_precision)
        U = count_unknowns(eul, M)
        if verbose:
            print(f"[M1] accuracy={accuracy}: M={M} terms, pi(M)={prime_pi(M)}, "
                  f"unknowns U={U}, weights needed ~{(U + 8 + 1 + 1)//2}")

        # --- M0+M1: oracle coefficients satisfy the relation (entire form) -----
        if M > 40:
            if verbose:
                print(f"     (skipping relation check: M={M} needs primes>37; "
                      f"lower accuracy for the data we have)")
            rel_err = mpf(0)
        else:
            mu = land.mu_from_point(target.point)
            b = eul.bcoeffs_from_ap(target.ap, M)
            errs = []
            for (m, al, be) in weights.weights[:4]:
                r = coefficient_relation(FIXED_S, mu=mu, nu=[], N=land.conductor,
                                         epsilon=target.epsilon, poles=[], g_m=m,
                                         g_alpha=al, g_beta=be, accuracy=accuracy,
                                         working_precision=working_precision, num_terms=M)
                errs.append(abs(r.evaluate(b)))
            rel_err = max(errs)
        results.append(("oracle coeffs satisfy coefficient_relation", rel_err,
                        mpf(10) ** (-(accuracy - 3)) if M <= 40 else mpf(1)))
        if verbose and M <= 40:
            print(f"[M0+M1] |coefficient_relation(true coeffs)| (4 weights) = "
                  f"{mpmath.nstr(rel_err,3)}")
    finally:
        mp.dps = old

    ok = all(err < thr for (_, err, thr) in results)
    print()
    print("LSEARCH M0+M1 SELFTEST", "PASS" if ok else "FAIL")
    for name, err, thr in results:
        print(f"   {'ok ' if err < thr else 'BAD'} {name:44s} err={mpmath.nstr(err,3):10s} (<{mpmath.nstr(thr,2)})")
    return ok


def selftest_m2(accuracy=8, working_precision=30, verbose=True):
    """M2 gate: the assembled residual system vanishes at the true unknowns, and a
    conditioning report for the solve subsystem."""
    results = []
    old = mp.dps
    mp.dps = working_precision
    try:
        target = gl3_known_target()
        system = build_equation_system(target.landscape, target.euler, target.point,
                                       accuracy, working_precision)
        primes = system.primes
        ap = {p: target.ap[p] for p in primes}
        u_true = pack_unknowns(target.epsilon, ap, primes)

        r = residual(system, u_true)
        max_all = max(abs(x) for x in r)
        max_det = max(abs(r[i]) for i in system.detector_idx)
        norm_res = abs(r[2 * len(system.A)])
        if verbose:
            print(f"[M2] M={system.M}, primes={len(primes)}, U={system.n_unknowns}, "
                  f"weights={len(system.A)}, equations={len(r)}")
            print(f"     |solve idx|={len(system.solve_idx)} (=U?), "
                  f"|detectors|={len(system.detector_idx)}")
            print(f"     residual at true unknowns:  max(all)={mpmath.nstr(max_all,3)}, "
                  f"max(detectors)={mpmath.nstr(max_det,3)}, |norm|={mpmath.nstr(norm_res,3)}")
        results.append(("residual = 0 at true unknowns", max_all,
                        mpf(10) ** (-(accuracy - 3))))
        results.append(("detectors = 0 at true unknowns", max_det,
                        mpf(10) ** (-(accuracy - 3))))

        # conditioning of the U x U solve subsystem at the true point
        J = jacobian(system, u_true)
        if len(J) != len(r):
            pass
        Jsolve = [J[i] for i in system.solve_idx]
        cond = _condition_number(Jsolve)
        sc = system.scales
        if verbose:
            print(f"     scale-factor spread: max/min = "
                  f"{mpmath.nstr(max(sc) / min(sc), 4)}")
            print(f"     solve-subsystem condition number = {mpmath.nstr(cond, 5)}")
        # not a pass/fail gate, but flag if hopeless
        results.append(("solve subsystem full rank / finite cond",
                        mpf(1) / cond if cond > 0 else mpf(0), mpf(1)))  # cond finite -> 1/cond>0
    finally:
        mp.dps = old

    ok = all(err < thr for (_, err, thr) in results[:2])
    print()
    print("LSEARCH M2 SELFTEST", "PASS" if ok else "FAIL")
    for name, err, thr in results:
        print(f"   {'ok ' if err < thr else '?? '} {name:42s} val={mpmath.nstr(err,3)}")
    return ok


def selftest_m3(accuracy=8, working_precision=30, verbose=True):
    """M3: the solver recovers the true coefficients at the true point (from a near
    guess) with detectors at the accuracy floor, and the detectors stay large at a
    wrong spectral point and for a spurious (blind-guess) solution."""
    results = []
    old = mp.dps
    mp.dps = working_precision
    try:
        target = gl3_known_target()
        system = build_equation_system(target.landscape, target.euler, target.point,
                                       accuracy, working_precision)
        gp = {p: target.ap[p] + mpc("1e-3", "1e-3") for p in system.primes}

        sol = solve_at_point(system, guess=gp)
        a2_err = abs(sol["ap"][2] - target.ap[2]) if sol else mpf(1)
        det_true = sol["det_res"] if sol else mpf(1)
        if verbose:
            print(f"[M3] true point, near guess: k={sol['k']}, det={mpmath.nstr(det_true,3)}, "
                  f"eps={mpmath.nstr(sol['epsilon'],8)}, a_2 err={mpmath.nstr(a2_err,3)}")
        results.append(("recover a_2 at true point", a2_err, mpf(10) ** (-(accuracy - 2))))
        results.append(("detectors reach floor at true point", det_true,
                        mpf(10) ** (-(accuracy - 2))))

        offpt = (target.point[0] + mpf("0.02"), target.point[1])
        soff = build_equation_system(target.landscape, target.euler, offpt,
                                     accuracy, working_precision)
        sol_off = solve_at_point(soff, guess=gp)
        det_off = sol_off["det_res"] if sol_off else mpf(0)
        if verbose:
            print(f"[M3] off point (lambda1+0.02): best det={mpmath.nstr(det_off,3)} "
                  f"(must stay >> floor)")
        # PASS if the off-point detectors are far above the true-point floor
        results.append(("detectors reject off point", mpf(10) ** (-3) / det_off
                        if det_off > 0 else mpf(0), mpf(1)))
    finally:
        mp.dps = old

    ok = all(err < thr for (_, err, thr) in results)
    print()
    print("LSEARCH M3 SELFTEST", "PASS" if ok else "FAIL")
    for name, err, thr in results:
        print(f"   {'ok ' if err < thr else 'BAD'} {name:40s} val={mpmath.nstr(err,3)}")
    return ok


def selftest_m4(accuracy=8, working_precision=30, verbose=True):
    """M4: from an off-centre triangular box the detector-line cloud concentrates
    much closer to the true point than the starting corner."""
    results = []
    old = mp.dps
    mp.dps = working_precision
    try:
        target = gl3_known_target()
        t1, t2 = target.point
        off = mpf("4e-4")
        P = (t1 - off, t2 - off)
        res = box_step(target.landscape, target.euler, P, mpf("1e-3"),
                       accuracy, working_precision, guess=target.ap, eps_guess=mpc(1),
                       verbose=verbose)
        if res is None:
            results.append(("box_step produced a cloud", mpf(1), mpf(0)))
        else:
            center, spread = cloud_center_spread(res["cloud"])
            err = max(abs(center[0] - t1), abs(center[1] - t2))
            if verbose:
                print(f"[M4] |cloud center - truth| = {mpmath.nstr(err,3)}  "
                      f"(start |P-truth| = {mpmath.nstr(off,2)}), spread = {mpmath.nstr(spread,3)}")
            # cloud center must be well inside the starting offset (search makes progress)
            results.append(("cloud center beats starting box", err, off / 4))
    finally:
        mp.dps = old
    ok = all(e < t for (_, e, t) in results)
    print()
    print("LSEARCH M4 SELFTEST", "PASS" if ok else "FAIL")
    for name, e, t in results:
        print(f"   {'ok ' if e < t else 'BAD'} {name:40s} val={mpmath.nstr(e,3)}")
    return ok


def _search_report(res, land, a, elapsed):
    """Build the human-readable result block for a command-line search: classify the
    outcome (wandered / box-did-not-decrease / partial / success), report the running
    time, and -- for a partial or full success -- the precision, accuracy, recovered
    coefficients, and a ready-to-run command to refine the result further."""
    ln10 = mpmath.log(10)

    def nstr(x, n):
        return mpmath.nstr(x, n)

    def digits(x):
        return max(0, int(-mpmath.log(x) / ln10)) if (x and x > 0) else 0

    init_box = mpf(a.boxsize)
    target = mpf(a.target)
    status = res.get("status")
    reason = res.get("reason", "")
    box = res.get("box")
    pt = res.get("point")

    out = ["=" * 68]
    out.append("SEARCH RESULT   landscape: %s" % land.name)
    out.append("start point: %s    initial box: %s    target: %s"
               % (a.point, nstr(init_box, 2), nstr(target, 2)))
    out.append("running time: %.1f s" % elapsed)
    if pt is not None:
        out.append("point found: l1 = %s ,  l2 = %s" % (nstr(pt[0], 18), nstr(pt[1], 18)))

    # ---- classify the outcome ------------------------------------------------
    if status == "success":
        kind = "success"
        out.append("OUTCOME: SUCCESS -- the box reached the target (%s)." % nstr(target, 2))
    elif status == "fail" and reason == "wandered":
        kind = "wandered"
        out.append("OUTCOME: FAILED (wandered) -- the centre drifted more than %s "
                   "from the start point." % nstr(mpf(a.wander_dist), 2))
    elif status == "fail" and reason == "no solution":
        kind = "nosol"
        out.append("OUTCOME: FAILED (no solution) -- could not solve at the start point.")
    elif status == "timeout":
        if box is not None and box < init_box * mpf("0.999"):
            kind = "partial"
            out.append("OUTCOME: TIMED OUT (partial) -- hit the time limit; the box had "
                       "shrunk from %s to %s (target %s not reached)."
                       % (nstr(init_box, 2), nstr(box, 2), nstr(target, 2)))
        else:
            kind = "timeout"
            out.append("OUTCOME: TIMED OUT -- hit the time limit with no decrease in the "
                       "box (no L-function resolved near the start point).")
    elif box is not None and box < init_box * mpf("0.999"):
        kind = "partial"
        out.append("OUTCOME: PARTIALLY SUCCESSFUL -- the box shrank from %s to %s "
                   "but did not reach the target %s."
                   % (nstr(init_box, 2), nstr(box, 2), nstr(target, 2)))
    else:
        kind = "noprogress"
        out.append("OUTCOME: FAILED (box did not decrease) -- no L-function was "
                   "resolved near the start point.")

    # ---- precision / accuracy + refine command (success or partial) ----------
    if kind in ("success", "partial"):
        out.append("spectral parameters determined to +- %s  (about %d digits)."
                   % (nstr(box, 2), digits(box)))
        cp = res.get("cloud_prec")
        if cp is not None:
            out.append("numerical precision of the cloud points: %s." % nstr(cp, 2))
        out.append("accuracy used: %d digits;  working precision: %d digits."
                   % (res.get("accuracy", 0), res.get("wp", 0)))
        cfr = res.get("coeff_fit_res")
        if cfr is not None:
            out.append("coefficient least-squares fit residual: %s." % nstr(cfr, 2))
        sol = res.get("sol")
        if sol:
            out.append("sign epsilon = %s" % nstr(sol["epsilon"], 12))
            for pp in sol["primes"]:
                out.append("  a_%-3d = %s" % (pp, nstr(sol["ap"][pp], 12)))
        # finer target: push 3 orders past a success, retry the original on a partial
        r_target = box * mpf("1e-3") if kind == "success" else target
        r_wp = res.get("wp", a.working_precision) + 10
        r_acc = res.get("accuracy", a.accuracy)
        out.append("")
        out.append("To refine further, run:")
        out.append("  python3 lsearch.py search --point=%s,%s --conductor %d \\"
                   % (nstr(pt[0], 20), nstr(pt[1], 20), land.conductor))
        out.append("    --boxsize %s --accuracy %d --working-precision %d \\"
                   % (nstr(box, 4), r_acc, r_wp))
        out.append("    --target %s --epsilon 1 --max-iter 12" % nstr(r_target, 2))

    out.append("=" * 68)
    return "\n".join(out)


def _search_cli(argv):
    """Command-line search of the degree-3, tempered, conductor-N GL(3) Maass-form
    landscape with Gamma factors
        Gamma_R(s + i*l1) Gamma_R(s + i*l2) Gamma_R(s - i*(l1+l2)),
    cold-started (Euler coefficients unknown) from a given spectral point."""
    import argparse
    import time
    p = argparse.ArgumentParser(
        prog="lsearch.py search",
        description="Search the degree-3, tempered, conductor-N GL(3) landscape "
                    "[Gamma_R(s+i*l1) Gamma_R(s+i*l2) Gamma_R(s-i*(l1+l2))] for an "
                    "L-function near a starting spectral point (coefficients unknown).")
    p.add_argument("--point", required=True,
                   help="starting spectral point 'l1,l2', e.g. '14.14,2.4'")
    p.add_argument("--conductor", type=int, default=1, help="conductor N")
    p.add_argument("--boxsize", default="1e-3", help="initial box half-width")
    p.add_argument("--accuracy", type=int, default=8, help="starting accuracy (digits)")
    p.add_argument("--working-precision", type=int, default=30,
                   help="starting mpmath precision (digits)")
    p.add_argument("--target", default="1e-6",
                   help="stop once the box half-width reaches this")
    p.add_argument("--max-iter", type=int, default=12)
    p.add_argument("--epsilon", default="1",
                   help="sign (root number) guess, e.g. '1' or '0.6+0.8j'")
    p.add_argument("--wander-dist", default="0.25",
                   help="abort if the centre drifts this far (absolute) from the start")
    p.add_argument("--timeout", type=float, default=600,
                   help="wall-clock seconds before giving up if no solution found (default 600)")
    p.add_argument("--append", default=None, metavar="FILE",
                   help="append the result report to this file (e.g. to log a grid of searches)")
    a = p.parse_args(argv)

    l1, l2 = (mpf(t.strip()) for t in a.point.split(","))
    mp.dps = a.working_precision
    land = Landscape(
        name="GL(3) degree 3, conductor %d" % a.conductor,
        degree=3, conductor=a.conductor, dim=2,
        mu_from_point=lambda pt: [1j * mpf(pt[0]), 1j * mpf(pt[1]),
                                  -1j * (mpf(pt[0]) + mpf(pt[1]))],
        nu_from_point=lambda pt: [])
    t0 = time.time()
    res = search(land, GL3_EULER, (l1, l2), mpf(a.boxsize), a.accuracy,
                 a.working_precision, mpf(a.target), guess=None,
                 eps_guess=mpc(complex(a.epsilon.replace(" ", ""))),
                 max_iter=a.max_iter, wander_dist=mpf(a.wander_dist),
                 timeout=a.timeout, verbose=True)
    elapsed = time.time() - t0

    report = _search_report(res, land, a, elapsed)
    print()
    print(report)
    if a.append:
        with open(a.append, "a") as fh:
            fh.write(report + "\n\n")
        print("(appended to %s)" % a.append)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        _search_cli(sys.argv[2:])
        raise SystemExit(0)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    acc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if which in ("all", "m01"):
        selftest(accuracy=acc)
        print()
    if which in ("all", "m2"):
        selftest_m2(accuracy=acc)
        print()
    if which in ("all", "m3"):
        selftest_m3(accuracy=acc)
        print()
    if which in ("all", "m4"):
        selftest_m4(accuracy=acc)
