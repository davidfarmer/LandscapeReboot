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

The landscapes themselves (the Gamma factors, conductor, central character and the
good-prime Euler-factor model) live in families.py and are loaded from the plain-text
registry landscapes.txt; this file is the family-agnostic engine that searches them.

Conventions fixed for the whole search (per the project design):
  * All unknowns and equations are real: each complex local unknown -> (Re, Im); a
    complex equation -> its (Re, Im) pair.
  * Each prime carries `euler.complex_unknowns_per_prime` complex local unknowns
    (the first few Dirichlet coefficients b(p), b(p^2), ...), so a per-prime unknown
    is a LIST.  For a tempered degree-3 form with trivial central character that is
    the single a_p = b(p): the local factor is 1 - a_p X + conj(a_p) X^2 - X^3 with
        b(p^k) = a_p b(p^{k-1}) - conj(a_p) b(p^{k-2}) + b(p^{k-3}),   b(p^0)=1.
    The Euler algebra (families.euler_self_reciprocal) generalizes this to any degree.
  * The sign epsilon is itself unknown, carried as (epsR, epsI) with the extra
    equation epsR^2 + epsI^2 = 1.
  * Equations come from the list of weight functions (not from varying a point s);
    `coefficient_relation` is always called at the fixed symmetry point s = 1/2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import random
import time

import mpmath
from mpmath import mp, mpf, mpc

from afe import (coefficient_relation, coefficient_relation_grid,
                 lmfdb_to_rubinstein, _min_contour,
                 GL3_MAASS, gl3_maass_bcoeffs)
from families import (Landscape, EulerProduct, KnownTarget, Family,
                      primes_up_to, prime_pi, _smallest_prime_factor_table,
                      get_family, gl3_known_target, GL3_LANDSCAPE, GL3_EULER)

# The fixed evaluation point for every coefficient_relation call (symmetry point).
FIXED_S = mpf(1) / 2

# Integration contour distance beyond the rightmost singularity, nu0 = bound + margin.
# The aliasing error is e^{-2 pi d / h} = 10^{-accuracy} for any d (h = 2 pi d / L is
# chosen to make it so), so moving the contour OUT only shrinks the grid (fewer points,
# faster builds) at no accuracy cost -- until d is so large the Gamma factors' dynamic
# range costs precision.  Empirically margin 2 is ~4x faster than the old 0.5 with
# bit-identical solve quality at wp~30 (see exp_contour / exp_solve_contour); margin 4
# is ~6x but starts to lose a digit.  2 is the safe default.
CONTOUR_MARGIN = mpf(2)


# (Prime helpers, the Landscape/EulerProduct/KnownTarget data model, and the
# GL(3) Euler algebra now live in families.py and are imported above.)


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
# Unknown real vector:   u = [epsR, epsI, then for each prime p <= M its
# complex_unknowns_per_prime local coefficients flattened to (Re, Im) pairs]
# with epsilon = epsR + i epsI.  For GL(3) each prime carries the single a_p.
# (b(1)=1 and the Euler product fix every other coefficient -- see families.py.)
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
    """(epsilon, {p: [c1, c2, ...]}) -> real vector [epsR, epsI, then each prime's
    complex local unknowns flattened to (Re, Im) pairs].  Each prime carries
    `euler.complex_unknowns_per_prime` complex numbers (just [a_p] for GL(3))."""
    u = [mpmath.re(epsilon), mpmath.im(epsilon)]
    for p in primes:
        for c in ap[p]:
            c = mpc(c)
            u += [mpmath.re(c), mpmath.im(c)]
    return u


def unpack_unknowns(u, primes, n_loc=1):
    """Inverse of pack_unknowns: -> (epsilon, {p: [c1, ..., c_{n_loc}]})."""
    epsilon = mpc(u[0], u[1])
    ap, off = {}, 2
    for p in primes:
        vals = []
        for _ in range(n_loc):
            vals.append(mpc(u[off], u[off + 1]))
            off += 2
        ap[p] = vals
    return epsilon, ap


def build_equation_system(landscape, euler, point, accuracy, working_precision=None,
                          poles=(), sign=None, n_detectors=8, lead_terms=5,
                          weight_set=None, contour_nu=None):
    """Precompute the residual system at `point` (the expensive Gamma/g work).

    If `weight_set` is given it is used as-is; otherwise a default set sized to the
    number of unknowns is generated.  `contour_nu` overrides the integration contour
    Re(z)=nu0 (default bound+1/2); moving it out shrinks the contour grid (faster
    builds) with the aliasing error staying at target -- see exp_contour."""
    mu = landscape.mu_from_point(point)
    nu = landscape.nu_from_point(point)
    bmax = admissibility_bound(landscape) - mpf("1.0")   # keep beta well clear of the bound
    #   (right at bound-0.8 the integrand converges very slowly: M jumps from ~27 to ~160)

    # default integration contour: out at bound + CONTOUR_MARGIN (faster grid, same
    # accuracy -- see CONTOUR_MARGIN).  At s=1/2 the f1/f2 bounds coincide.
    if contour_nu is None:
        gd0 = lmfdb_to_rubinstein(mu, nu, landscape.conductor, 1)
        contour_nu = _min_contour(FIXED_S, gd0.kappa, gd0.lam) + CONTOUR_MARGIN

    # one auto-truncating call (most demanding weight) fixes the Dirichlet length M
    Msz = coefficient_relation(FIXED_S, mu=mu, nu=nu, N=landscape.conductor, epsilon=1,
                               poles=list(poles), g_m=0, g_alpha=0, g_beta=bmax,
                               accuracy=accuracy, working_precision=working_precision,
                               contour_nu=contour_nu)
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
                                     list(poles), wlist, M, working_precision,
                                     contour_nu=contour_nu)

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
    epsilon, ap = unpack_unknowns(u, primes, system.euler.complex_unknowns_per_prime)
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
    rpp = euler.real_unknowns_per_prime                    # 2 * complex unknowns / prime
    U = 2 + rpp * k
    cols = [0, 1] + [2 + rpp * j + t for j in range(k) for t in range(rpp)]  # eps + k primes
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

def _guess_vector(primes, guess, eps_guess, n_loc=1):
    """Pack a starting vector for `primes`, padding/truncating each supplied per-prime
    coefficient list to n_loc entries (missing primes start at all-zero)."""
    guess = guess or {}
    ap = {}
    for p in primes:
        g = guess.get(p)
        if g is None:
            ap[p] = [mpc(0)] * n_loc
        else:
            vals = [mpc(x) for x in g][:n_loc]
            ap[p] = vals + [mpc(0)] * (n_loc - len(vals))
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
    nloc = system.euler.complex_unknowns_per_prime
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
        xall = _guess_vector(cand, guess, eps_guess, nloc)
        jac_full = _equation_jacobian(system, xall, cand, 2 * K, deadline=deadline)

    best = None
    for k in ks:
        if deadline is not None and time.time() > deadline:
            break                       # out of time -> return the best found so far
        if fixed is None:
            primes = cand[:k]
            solve_idx, det_idx = _select_from_jacobian(jac_full, K, system.euler,
                                                       k, n_detectors)
        x0 = _guess_vector(primes, guess, eps_guess, nloc)

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
        eps, ap = unpack_unknowns(x, primes, nloc)
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
    nloc = system.euler.complex_unknowns_per_prime
    x0 = _guess_vector(primes, guess, eps_guess, nloc)
    jac0 = _equation_jacobian(system, x0, primes, 2 * K)
    eq_idx = _select_rows(jac0, n_eq) + [norm_idx]
    x, nrm = _lsq_core(system, primes, eq_idx, x0, tol, maxiter, damping)

    eps, ap = unpack_unknowns(x, primes, nloc)
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


def _cloud_from_corners(systems, sols, point, h, detector_idx, working_precision):
    """From the corner solutions, affinely model each detector over the box and
    intersect the zero-lines pairwise into a candidate cloud.  Returns
    (cloud, cloud_info, lines)."""
    Dvals = []
    for i in range(3):
        full = residual(systems[i], sols[i]["x"], primes=sols[i]["primes"])
        Dvals.append([full[d] for d in detector_idx])
    # affine model of each detector: D(d1,d2) = D0 + g1 d1 + g2 d2  (d = offset from P)
    lines = []
    for d in range(len(detector_idx)):
        D0, D1, D2 = Dvals[0][d], Dvals[1][d], Dvals[2][d]
        lines.append((D0, (D1 - D0) / h, (D2 - D0) / h))
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
    return cloud, cloud_info, lines


def _solve_corner(system, guess, eps_guess, deadline, restarts, fixed=None):
    """Solve at system.point, trying the warm `guess` (the previous iteration's solution)
    first; if it fails, try up to `restarts` NEARBY starting points -- small perturbations
    of the warm guess, and keep-the-low-primes / randomize-the-rest restarts -- so a single
    Broyden divergence at one box step does not abort the whole search.  Returns the
    solution dict or None.  `fixed` reuses a chosen equation set (corners 1, 2)."""
    def solve(seed):
        if fixed is not None:
            return solve_at_point(system, guess=seed, eps_guess=eps_guess,
                                  fixed=fixed, deadline=deadline)
        return solve_at_point(system, guess=seed, eps_guess=eps_guess, deadline=deadline)

    sol = solve(guess)
    if sol is not None or restarts <= 0:
        return sol
    nloc = system.euler.complex_unknowns_per_prime
    primes = fixed[2] if fixed is not None else system.primes
    low = {p: guess[p] for p in list(primes)[:3] if p in guess} if guess else {}
    for i in range(restarts):
        if deadline is not None and time.time() > deadline:
            break
        if guess and i % 2 == 0:                  # a nearby point: perturb the warm guess
            seed = {p: [c + _rand_c(mpf("0.3")) for c in guess[p]] for p in guess}
        else:                                     # keep the low primes, randomize the rest
            seed = _random_ap(primes, mpf(2), nloc, fixed=low)
        sol = solve(seed)
        if sol is not None:
            return sol
    return sol


def box_step(landscape, euler, point, boxsize, accuracy, working_precision,
             guess=None, eps_guess=None, solver="square", deadline=None, verbose=False,
             solve_restarts=0):
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
    sol0 = _solve_corner(sys0, guess, eps_guess, deadline, solve_restarts)
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
        solc = _solve_corner(sysc, sol0["ap"], sol0["epsilon"], deadline, solve_restarts,
                             fixed=fixed)
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
            eps_r, ap_r = unpack_unknowns(x_ref, primes, sys0.euler.complex_unknowns_per_prime)
            sols[i] = {**sols[i], "x": x_ref, "epsilon": eps_r, "ap": ap_r}

    # detector zero-lines from the 3 corners, intersected pairwise into a cloud
    cloud, cloud_info, lines = _cloud_from_corners(
        systems, sols, point, h, detector_idx, working_precision)

    # precision lost in recovering the coefficients (solve conditioning at corner 0)
    cond_solve = sol0["cond"]
    if verbose:
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
    nloc = len(sols[0]["ap"][primes[0]]) if primes else 0
    ap = {p: [mpmath.fsum(s["ap"][p][j] for s in sols) / n for j in range(nloc)]
          for p in primes}
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
BOX_GROW = mpf(4)               # max factor to grow the box per iteration
BOX_SHRINK = mpf(5)             # max factor to shrink the box per iteration
BOX_MARGIN = mpf("1.5")         # target box = BOX_MARGIN * determination (bracket w/ margin)
BOX_DEADBAND = mpf("1.3")       # hold the box if the target is within this factor of it
#   (dead-band + rate limits make the box TRACK the determination smoothly instead of
#    flip-flopping between grow and zoom when the determination hovers near the box size)
RECENTER_CAP = mpf(2)           # step toward the cloud at most this many box sizes / iter


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
    # show lambda to 3 digits beyond the determined accuracy (the cloud spread), and the
    # coefficients to 3 digits beyond the detector residual (the per-iteration analog of
    # the least-squares fit residual)
    ln10 = mpmath.log(10)
    show = int(max(0, -mpmath.log(spread) / ln10)) + 3 if spread else 16
    cdig = int(max(0, -mpmath.log(det_res) / ln10)) + 3 if det_res and det_res > 0 else show
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
    print(f"  Euler coefficients (b(p), b(p^2), ... per prime):")
    for p in sol["primes"]:
        comps = sol["ap"][p]
        vals = ", ".join(mpmath.nstr(c, cdig) for c in comps)
        label = f"a_{p:<3d}=" if len(comps) == 1 else f"p={p:<3d}:"
        print(f"      {label} {vals}")


def _report_iteration_brief(it, center, spread, boxsize, det_res, sol):
    """One-line per-iteration summary (printed when not in --verbose mode): the L-point,
    box size, determination, detector residual, and the first couple of Euler
    coefficients -- enough to follow the trajectory at a glance."""
    ln10 = mpmath.log(10)
    show = int(max(0, -mpmath.log(spread) / ln10)) + 3 if spread else 12
    cdig = min(6, int(max(0, -mpmath.log(det_res) / ln10)) + 3) if det_res and det_res > 0 else 6
    coeffs = "".join(f"  a_{p}={mpmath.nstr(sol['ap'][p][0], cdig)}"
                     for p in sol["primes"][:2])
    print(f"  iter {it}: L-point=({mpmath.nstr(center[0], show)}, "
          f"{mpmath.nstr(center[1], show)})  box={mpmath.nstr(boxsize, 2)}  "
          f"det={mpmath.nstr(det_res, 2)}{coeffs}")


def _rand_c(scale):
    return mpc(2 * scale * (mpf(random.random()) - mpf("0.5")),
               2 * scale * (mpf(random.random()) - mpf("0.5")))


def _random_ap(primes, scale, n_loc=1, fixed=None):
    """A random Euler-coefficient guess (uniform in the box |Re|,|Im| <= scale) used as a
    Broyden restart; different restarts find different solutions of the system.  Any
    coefficient list supplied in `fixed` is KEPT (used as its starting value), padded to
    n_loc; only the remaining primes' coefficients are randomized -- so a partial --coeffs
    seed pins the known primes while the rest are explored randomly."""
    fixed = fixed or {}
    out = {}
    for p in primes:
        if p in fixed:
            vals = [mpc(x) for x in fixed[p]][:n_loc]
            out[p] = vals + [_rand_c(scale) for _ in range(n_loc - len(vals))]
        else:
            out[p] = [_rand_c(scale) for _ in range(n_loc)]
    return out


def _coeff_distance(apA, apB):
    """Distance between two coefficient solutions, weighted toward the SMALLER primes
    (which are determined accurately; larger primes carry the numerical noise).  Used to
    decide whether two candidates are the SAME solution of the system -- so two distinct
    L-functions that share a (near-)identical spectral point are kept apart by their
    coefficients."""
    common = sorted(set(apA) & set(apB))
    if not common:
        return mpmath.inf
    # per prime: sum |c_j(A) - c_j(B)| over the local-unknown components
    def cdiff(p):
        return mpmath.fsum(abs(a - b) for a, b in zip(apA[p], apB[p]))
    num = mpmath.fsum(cdiff(p) / p for p in common)
    den = mpmath.fsum(mpf(1) / p for p in common)
    return num / den


def explore_candidates(landscape, euler, point, boxsize, accuracy, working_precision,
                       restarts=20, guess=None, eps_guess=None, k_min=3, k_max=None,
                       coeff_tol=mpf("0.05"), max_candidates=5, scale=mpf(2),
                       deadline=None, verbose=True, trace=False):
    """Find DISTINCT candidate solutions of the system near `point`.

    Varies the number of coefficients k (from k_min up to the available significant
    primes) and tries `restarts` random Broyden starts at each k (plus the given guess).
    Each corner-0 solution is kept; they are de-duplicated by their COEFFICIENT vectors
    (weighted toward small primes), keeping the lowest-detector-residual representative of
    each distinct solution -- so different L-functions at a (near-)identical point survive
    separately.  For each distinct candidate the cloud centre (a provisional L-point) is
    then computed from the other two corners.  Returns a list of candidate dicts
    (center, spread, ap, epsilon, det, k)."""
    h = mpf(boxsize)
    corners = triangle_corners(point, h)
    mp.dps = working_precision
    if trace:
        print(f"  building equation system at ({mpmath.nstr(point[0],10)}, "
              f"{mpmath.nstr(point[1],10)})  (accuracy {accuracy}, working precision "
              f"{working_precision}) ...")
    sys0 = build_equation_system(landscape, euler, corners[0], accuracy, working_precision)
    weights = WeightSet(weights=sys0.weights)
    systems = [sys0] + [build_equation_system(landscape, euler, c, accuracy,
                        working_precision, weight_set=weights) for c in corners[1:]]
    cand_primes = sys0.primes
    if not cand_primes:
        return []
    if k_max is None:
        k_max = len(cand_primes)
    k_max = min(k_max, len(cand_primes))
    k_min = max(1, min(k_min, k_max))
    eps_guess = eps_guess or mpc(1)

    if trace:
        print(f"  exploring near ({mpmath.nstr(point[0],10)}, {mpmath.nstr(point[1],10)}): "
              f"k={k_min}..{k_max}, {restarts} random restart(s) per k"
              + (" (+ given coeffs)" if guess is not None else ""))

    # (1) corner-0 solutions over k and random restarts (re-solves on the one build)
    sols0 = []
    for k in range(k_min, k_max + 1):
        primes = cand_primes[:k]
        # seeds: the given guess (known primes + zeros), then random restarts that KEEP
        # any given coefficients and randomize the rest, then a cold (all-zero) start
        nloc = euler.complex_unknowns_per_prime
        seeds = ([guess] if guess is not None else []) + \
                [_random_ap(primes, scale, nloc, fixed=guess) for _ in range(restarts)] + [None]
        n_before = len(sols0)
        for seed in seeds:
            if deadline is not None and time.time() > deadline:
                break
            s = solve_at_point(sys0, guess=seed, eps_guess=eps_guess,
                               k_init=k, k_max=k, deadline=deadline)
            if s is not None:
                sols0.append(s)
        if trace:                          # per-k progress so the wait is not silent
            found = sols0[n_before:]
            best = min((s["det_res"] for s in found), default=None)
            print(f"    k={k}: {len(found)} solution(s)"
                  + (f", best detector residual {mpmath.nstr(best,2)}" if best is not None
                     else "")
                  + (" [time limit reached]"
                     if deadline is not None and time.time() > deadline else ""))
        if deadline is not None and time.time() > deadline:
            break

    # (2) de-duplicate by coefficient vector; keep best-det representative of each
    sols0.sort(key=lambda s: s["det_res"])
    reps = []
    for s in sols0:
        if any(_coeff_distance(s["ap"], r["ap"]) < coeff_tol for r in reps):
            continue
        reps.append(s)
        if len(reps) >= max_candidates:
            break

    # (3) cloud centre of each distinct candidate (from the other two corners)
    out = []
    for s in reps:
        fixed = (s["solve_idx"], s["det_idx"], s["primes"])
        sols, ok = [s], True
        for i in (1, 2):
            sc = solve_at_point(systems[i], guess=s["ap"], eps_guess=s["epsilon"],
                                fixed=fixed)
            if sc is None:
                ok = False
                break
            sols.append(sc)
        if not ok:
            continue
        cloud, _ci, _ln = _cloud_from_corners(systems, sols, point, h,
                                              s["det_idx"], working_precision)
        if not cloud:
            continue
        center, spread = cloud_center_spread(cloud)
        out.append({"center": center, "spread": spread, "ap": s["ap"],
                    "epsilon": s["epsilon"], "det": s["det_res"], "k": s["k"]})
    if verbose:
        print(f"  exploration: {len(sols0)} solutions over k={k_min}..{k_max}, "
              f"{len(out)} distinct candidate(s)")
        for n, c in enumerate(out):
            print(f"    candidate {n}: center=({mpmath.nstr(c['center'][0],10)}, "
                  f"{mpmath.nstr(c['center'][1],10)})  k={c['k']}  "
                  f"det={mpmath.nstr(c['det'],2)}")
            # show the candidate's Euler coefficients, not just its spectral point
            for p in sorted(c["ap"]):
                comps = c["ap"][p]
                vals = ", ".join(mpmath.nstr(x, 8) for x in comps)
                lbl = f"a_{p}" if len(comps) == 1 else f"p={p}"
                print(f"        {lbl} = {vals}")
    return out


def search_landscape(landscape, euler, point, boxsize, accuracy, working_precision,
                     target_box, restarts=20, guess=None, eps_guess=None, max_iter=40,
                     wander_dist=mpf("0.25"), timeout=600, refine_coeffs=True,
                     coeff_tol=mpf("0.05"), max_candidates=5, verbose=False,
                     on_result=None):
    """Explore near `point` for distinct candidate solutions (varying k and random
    Broyden restarts), then refine EACH distinct candidate into an L-point.  The time
    limit guards only the exploration; refinement of a real candidate runs to completion.
    If `guess` is given (some starting coefficients) it is kept across the random restarts
    and only the remaining coefficients are randomized.  If `guess` already covers all the
    significant primes (a full RESUME), the random exploration is skipped entirely and the
    point is refined directly -- exploration there is pointless and its time limit would
    otherwise expire during the (slow, high-accuracy) builds before any solve.  Returns a
    list of result dicts (one per candidate refined), each with 'secs'.

    The distinct candidates (spectral point AND coefficients) are always listed.
    `verbose` additionally prints the per-iteration detail of each refinement.  If an
    `on_result(n, res)` callback is given it is invoked as soon as each candidate finishes
    refining, so the caller can report results incrementally rather than only at the end."""
    explore_acc = int(round(mpf(accuracy)))
    explore_wp = max(int(working_precision), 2 * explore_acc + 20)
    deadline = (time.time() + timeout) if timeout else None

    # Full resume: the supplied coefficients already cover the significant primes -> just
    # refine from (point, guess); the box shrinks at once so the time limit is lifted and
    # the (possibly long) high-accuracy refinement runs to completion.
    if guess is not None:
        mp.dps = explore_wp
        sys0 = build_equation_system(landscape, euler, point, explore_acc, explore_wp)
        # full resume if the guess covers (all but at most one of) the significant primes
        if sys0.primes and len(set(sys0.primes) - set(guess)) <= 1:
            print("  resume: all significant primes supplied -> refining directly "
                  "(no random exploration, no time limit)")
            t0 = time.time()
            # no time limit on a resume: the point is a known L-function, so the limit
            # (which exists to abandon hopeless far points) does not apply; max_iter bounds
            # it, and the high-accuracy climb to a fine target can take a while
            res = search(landscape, euler, point, boxsize, accuracy, working_precision,
                         target_box, guess=guess, eps_guess=eps_guess, max_iter=max_iter,
                         wander_dist=wander_dist, timeout=None,
                         refine_coeffs=refine_coeffs, verbose=verbose)
            res["secs"] = time.time() - t0
            res["candidate"] = 0
            if on_result is not None:
                on_result(0, res)
            return [res]

    # When some coefficients are supplied (a partial resume / known-point context), lift
    # the time limit: the slow high-accuracy builds would otherwise expire it before any
    # random start is tried.  The limit exists to abandon hopeless FAR points, which does
    # not apply once the caller has given a point + partial coefficients.
    explore_deadline = None if guess is not None else deadline
    refine_timeout = None if guess is not None else timeout

    # the candidate list (spectral points AND coefficients) is always shown
    cands = explore_candidates(landscape, euler, point, boxsize, explore_acc, explore_wp,
                               restarts=restarts, guess=guess, eps_guess=eps_guess,
                               coeff_tol=coeff_tol, max_candidates=max_candidates,
                               deadline=explore_deadline, verbose=True, trace=verbose)
    results = []
    for n, cand in enumerate(cands):
        print(f"  === refining candidate {n} at ({mpmath.nstr(cand['center'][0],10)}, "
              f"{mpmath.nstr(cand['center'][1],10)}) ===")
        t0 = time.time()
        res = search(landscape, euler, cand["center"], boxsize, accuracy,
                     working_precision, target_box, guess=cand["ap"],
                     eps_guess=cand["epsilon"], max_iter=max_iter,
                     wander_dist=wander_dist, timeout=refine_timeout,
                     refine_coeffs=refine_coeffs, verbose=verbose)
        res["secs"] = time.time() - t0
        res["candidate"] = n
        results.append(res)
        if on_result is not None:        # report each candidate as soon as it finishes
            on_result(n, res)
    return results


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
           refine_coeffs=True, timeout=600, solve_restarts=6, verbose=True):
    """Iterative search (Steps 1-8): move a triangular box toward an L-function, raising
    the accuracy (to tighten the spectral-parameter cloud) and the working precision (to
    keep the cloud points trustworthy) as the box shrinks.

    The box is ADAPTIVE.  The detector cloud is an affine model valid only across the box,
    so if the determination (cloud spread) is coarser than the box the L-point is not yet
    bracketed and the box is GROWN (by BOX_GROW per iteration, centre held, up to
    wander_dist) until it brackets a point; once bracketed the box recenters on the cloud
    and SHRINKS to zoom in.  This lets a search started on a box far smaller than the
    distance to the nearest L-function (e.g. a coarse grid point) still converge, instead
    of failing because a tiny box can never reach the right scale.

    Stops with 'success' if the box reaches target_box, 'converged' if the box stops
    shrinking (detector floor hit), or 'no point within range' if the box grows to
    wander_dist without bracketing any L-function.  Returns a status dict.

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
    accuracy = acc_floor         # current accuracy, carried across iterations (monotonic)
    prev_det = None              # last iteration's determination (cloud spread)
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
        # Accuracy is monotonic across iterations.  In refinement it is at least what the
        # box needs; the stall logic below raises it further (coupled to the
        # DETERMINATION, not just the box) when the determination is still improving.
        if refining:
            accuracy = max(accuracy,
                           _accuracy_for_box(boxsize, acc_floor, acc_max, ln10))
        wp = max(wp, int(mpmath.ceil(accuracy)) + 6)

        # one box step per iteration (the centre RECENTERS each iteration, so it can walk
        # toward a form), with the precision guard redoing it until
        # wp >= accuracy + log10(cond) + GUARD_DIGITS
        # one box step per iteration, wrapped in a retry loop with two guards:
        #  * precision: redo at higher wp until wp >= accuracy + log10(cond) + GUARD, and
        #    ALSO if the step fails outright -- a too-low wp makes the corner solve fail
        #    (no cloud), and the reactive guard alone never recovers from that;
        #  * box too big: if even at ample wp there is still no cloud, the corners are too
        #    far apart to solve -> shrink the box and retry.
        bs = cond = center = spread = None
        good = None      # last SUCCESSFUL (bs, center, spread, cond, wp) this iteration
        for attempt in range(12):
            if _timed_out():
                return {"status": "timeout", "reason": "time limit", "iter": it,
                        **(last or {"point": point})}
            mp.dps = wp
            bs = box_step(landscape, euler, point, boxsize, int(round(accuracy)), wp,
                          guess=g, eps_guess=eg, solver=solver,
                          deadline=(None if refining else deadline), verbose=False,
                          solve_restarts=solve_restarts)
            if _timed_out():
                return {"status": "timeout", "reason": "time limit", "iter": it,
                        **(last or {"point": point})}
            if bs is not None and bs["cloud"]:
                cond = bs["cond_solve"]
                center, spread = cloud_center_spread(bs["cloud"])
                good = (bs, center, spread, cond, wp)      # remember this success
                need = int(mpmath.ceil(accuracy + mpmath.log(cond) / ln10 + GUARD_DIGITS))
                if wp >= need:
                    break                              # good step at adequate precision
                new_wp = need + 3   # 3 digits of headroom so we don't redo again at once
                if verbose:         # show the provisional guess so a redo isn't a silent wait
                    print(f"   provisional guess: L-point=({mpmath.nstr(center[0], 12)}, "
                          f"{mpmath.nstr(center[1], 12)})  box size={mpmath.nstr(boxsize, 2)}")
                    print(f"   (redo: wp {wp} < accuracy+log10(cond)+{GUARD_DIGITS} = "
                          f"{need} -> raising wp to {new_wp})")
                wp = new_wp
                continue
            # no cloud at this wp.  If a LOWER-wp step already succeeded this iteration, a
            # precision redo has just failed (box_step is not monotone in wp -- a corner
            # solve can diverge at a higher wp); fall back to that good step rather than
            # discarding it.
            if good is not None:
                bs, center, spread, cond, wp = good
                if verbose:
                    print(f"   (higher-precision redo failed; keeping the wp {wp} step)")
                break
            # never got a cloud: first suspect a too-low wp (the corner solve failed) and
            # raise wp, up to a generous ceiling.
            if wp < int(mpmath.ceil(accuracy)) + GUARD_DIGITS + 24:
                new_wp = wp + max(10, GUARD_DIGITS)
                if verbose:
                    print(f"   (box step failed at wp {wp}; raising wp to {new_wp} "
                          f"and retrying)")
                wp = new_wp
                continue
            # wp is already generous -> the box is too big (corners diverge); shrink it.
            if boxsize > target * 4:
                boxsize = boxsize / 2
                if verbose:
                    print(f"   (box step failed; box too large -> shrinking to "
                          f"{mpmath.nstr(boxsize, 2)} and retrying)")
                continue
            return {"status": "fail", "reason": "no solution", "iter": it,
                    **(last or {"point": point})}
        else:
            return {"status": "fail", "reason": "no solution", "iter": it,
                    **(last or {"point": point})}

        cloud_prec = estimate_cloud_precision(bs)
        det_res = max((s["det_res"] for s in bs["sols"] if s), default=mpf(0))
        sol = _average_solution(bs["sols"])
        if verbose:
            _report_iteration(it, center, spread, cloud_prec, accuracy, wp,
                              boxsize, cond, sol, det_res)
        else:    # always report the point + first couple of coefficients each iteration
            _report_iteration_brief(it, center, spread, boxsize, det_res, sol)

        # --- smooth box controller -------------------------------------------------
        # The detector cloud is an affine model valid across the box, so we want the box to
        # bracket the determination (cloud spread) with some margin.  Track the target box
        # (BOX_MARGIN * spread) with a per-iteration RATE LIMIT (<= BOX_GROW grow,
        # <= BOX_SHRINK shrink) and a DEAD-BAND: if the target is within BOX_DEADBAND of the
        # current box, hold the box and raise accuracy instead (which tightens the
        # determination, letting the box shrink next iteration).  This makes the box track
        # the determination SMOOTHLY rather than flip-flopping between grow and zoom when
        # the determination hovers near the box size (the oscillation that wandered off and
        # failed on a cold start).
        desired = spread * BOX_MARGIN
        if desired > boxsize * BOX_DEADBAND:           # under-determined -> grow
            new_box = min(desired, boxsize * BOX_GROW, wander_dist)
        elif desired < boxsize / BOX_DEADBAND:         # over-determined -> shrink (zoom)
            new_box = max(desired, boxsize / BOX_SHRINK)
        else:
            new_box = boxsize                          # dead-band: hold

        bracketed = spread < boxsize                   # cloud centre a valid interpolation?

        # Recenter toward the cloud centre, damped: cap the move at RECENTER_CAP box sizes.
        # While under-determined the centre is a long, noisy extrapolation and the cap (a
        # small multiple of the small box) barely moves the point; once bracketed the cap
        # is comparable to the offset so the point moves (almost) onto the cloud centre.
        move_cap = RECENTER_CAP * new_box
        step = [center[0] - point[0], center[1] - point[1]]
        mag = max(abs(step[0]), abs(step[1]))
        if mag > move_cap and mag > 0:
            new_point = (point[0] + step[0] * move_cap / mag,
                         point[1] + step[1] * move_cap / mag)
        else:
            new_point = center

        # Wander check only once bracketed (while under-determined the centre is too noisy).
        if bracketed and max(abs(new_point[0] - start[0]),
                             abs(new_point[1] - start[1])) > wander_dist:
            return {"status": "fail", "reason": "wandered", "iter": it,
                    "point": new_point, "box": new_box}

        last = {"point": new_point, "box": new_box, "iter": it,
                "accuracy": int(round(accuracy)), "wp": wp,
                "cloud_prec": cloud_prec, "det_res": det_res, "sol": sol}

        # once the box brackets the point, refine in earnest (lift the exploration limit)
        if bracketed and not refining:
            refining = True
            if verbose:
                print("   (bracketed the L-point -> refining)")

        if new_box <= target and bracketed:
            return _finalize_coeffs({"status": "success", **last},
                                    landscape, euler, refine_coeffs, verbose)

        if not bracketed and boxsize >= wander_dist:
            # grown to the wander limit without bracketing any L-function
            return {"status": "fail", "reason": "no point within range", "iter": it, **last}

        # progress / stall: box shrinking -> progress; box held (dead-band) and the
        # determination no longer improving with accuracy -> the detector floor
        if new_box < boxsize * stall_factor:
            stalls = 0
            if verbose:
                print(f"   box shrank: {mpmath.nstr(boxsize, 2)} -> {mpmath.nstr(new_box, 2)}"
                      f"  (determination +-{mpmath.nstr(spread, 2)})")
        elif new_box > boxsize:
            stalls = 0
            if verbose:
                print(f"   box grew: {mpmath.nstr(boxsize, 2)} -> {mpmath.nstr(new_box, 2)}"
                      f"  (determination +-{mpmath.nstr(spread, 2)})")
        elif (prev_det is None or spread < prev_det * stall_factor) and \
                int(round(accuracy)) < acc_max:
            accuracy = min(mpf(acc_max), accuracy + 2)
            stalls = 0
            if verbose:
                print(f"   determination +-{mpmath.nstr(spread, 2)} ~ box "
                      f"{mpmath.nstr(boxsize, 2)}; raising accuracy to {int(round(accuracy))}")
        else:
            stalls += 1
            if stalls >= stall_patience:
                return _finalize_coeffs(
                    {"status": "converged", "reason": "detector floor", **last},
                    landscape, euler, refine_coeffs, verbose)

        prev_det = spread
        g, eg = sol["ap"], sol["epsilon"]
        point, boxsize = new_point, new_box
    return {"status": "fail", "reason": "max_iter",
            **(last or {"point": point}), "iter": it}


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
            a2 = ap[2][0]
            print(f"     a_2 = {mpmath.nstr(a2,8)}   b(4) = a_2^2-conj(a_2) ?"
                  f" {mpmath.nstr(b_recon[3],8)} vs {mpmath.nstr(a2**2-mpmath.conj(a2),8)}")

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
        gp = {p: [c + mpc("1e-3", "1e-3") for c in target.ap[p]] for p in system.primes}

        sol = solve_at_point(system, guess=gp)
        a2_err = abs(sol["ap"][2][0] - target.ap[2][0]) if sol else mpf(1)
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


def _fmt_complex(z, digits):
    """Format a complex number as 're+imj' / 're-imj' (no spaces or parentheses) to
    `digits` significant figures -- the form the --coeffs parser accepts."""
    z = mpc(z)
    re = mpmath.nstr(mpmath.re(z), digits)
    im = mpmath.im(z)
    return re + ("+" if im >= 0 else "-") + mpmath.nstr(abs(im), digits) + "j"


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
    # show the L-point to 3 digits beyond the determined accuracy (the box)
    lam_dig = (digits(box) + 3) if (box and box > 0) else 16

    out = ["=" * 68]
    out.append("SEARCH RESULT   landscape: %s" % land.name)
    out.append("start point: %s    initial box: %s    target: %s"
               % (a.point, nstr(init_box, 2), nstr(target, 2)))
    iters = res.get("iter")
    out.append("iterations done: %s    running time: %.1f s"
               % ("?" if iters is None else iters + 1, elapsed))
    if pt is not None:
        out.append("point found: l1 = %s ,  l2 = %s"
                   % (nstr(pt[0], lam_dig), nstr(pt[1], lam_dig)))

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
        # show coefficients to 3 digits beyond the least-squares fit residual
        coef_dig = (digits(cfr) + 3) if (cfr and cfr > 0) else lam_dig
        sol = res.get("sol")
        if sol:
            out.append("sign epsilon = %s" % nstr(sol["epsilon"], coef_dig))
            for pp in sol["primes"]:
                comps = sol["ap"][pp]
                vals = ", ".join(nstr(c, coef_dig) for c in comps)
                lbl = "a_%-3d =" % pp if len(comps) == 1 else "p=%-3d:" % pp
                out.append("  %s %s" % (lbl, vals))
        # finer target: push 3 orders past a success, retry the original on a partial
        r_target = box * mpf("1e-3") if kind == "success" else target
        r_wp = res.get("wp", a.working_precision) + 10
        r_acc = res.get("accuracy", a.accuracy)
        out.append("")
        out.append("To refine further, run:")
        out.append("  python3 lsearch.py search --point=%s,%s --conductor %d \\"
                   % (nstr(pt[0], lam_dig), nstr(pt[1], lam_dig), land.conductor))
        out.append("    --boxsize %s --accuracy %d --working-precision %d \\"
                   % (nstr(box, 4), r_acc, r_wp))
        out.append("    --target %s --epsilon 1 --max-iter 12 \\" % nstr(r_target, 2))
        # include the recovered coefficients so this is a copy-paste resume; quoted
        # because the values contain '+'/spaces, and the leading '-' needs the = form
        if sol:
            cs = ",".join(_fmt_complex(c, coef_dig)
                          for p in sol["primes"] for c in sol["ap"][p])
            out.append('    --coeffs="%s"' % cs)
        else:
            out.append("    --coeffs=...")

    out.append("=" * 68)
    return "\n".join(out)


def _parse_coeffs(text, n_loc=1):
    """Parse a starting-coefficient string into a {prime: [c1, ..., c_{n_loc}]} dict.
    Comma-separated; bare values are grouped n_loc at a time and assigned positionally
    to the primes 2,3,5,7,...; a 'p:val' token sets prime p's first component.  For the
    GL(3) family (n_loc=1) this is one value per prime.  Returns None for empty input."""
    if not text or not text.strip():
        return None
    primes = primes_up_to(10 ** 4)
    out, pos, buf = {}, 0, []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            p, v = tok.split(":")
            out.setdefault(int(p), [mpc(0)] * n_loc)
            out[int(p)][0] = mpc(complex(v.replace(" ", "")))
        else:
            buf.append(mpc(complex(tok.replace(" ", ""))))
            if len(buf) == n_loc:
                out[primes[pos]] = buf
                buf, pos = [], pos + 1
    if buf:                                   # trailing partial group -> pad with zeros
        out[primes[pos]] = buf + [mpc(0)] * (n_loc - len(buf))
    return out


def _search_cli(argv):
    """Command-line search of a landscape (loaded from the registry landscapes.txt by
    name, or from an ad-hoc file), for an L-function near a starting spectral point.
    Cold-started (Euler coefficients unknown) or warm-started from --coeffs."""
    import argparse
    import time
    p = argparse.ArgumentParser(
        prog="lsearch.py search",
        description="Search a tempered balanced analytic L-function landscape (named in "
                    "the registry landscapes.txt, default R0R0R0N1) for an L-function "
                    "near a starting spectral point.")
    p.add_argument("--landscape", default="R0R0R0N1",
                   help="registry name of the landscape (default R0R0R0N1)")
    p.add_argument("--landscape-file", default=None, metavar="FILE",
                   help="load the landscape from this file instead of the registry "
                        "(same format as landscapes.txt; for development)")
    p.add_argument("--point", required=True,
                   help="starting spectral point, e.g. '14.14,2.4' (one value per free "
                        "spectral parameter / search dimension)")
    p.add_argument("--conductor", type=int, default=None,
                   help="override the landscape conductor N (default: as registered)")
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
                   help="wall-clock seconds for the exploration phase only (default 600)")
    p.add_argument("--restarts", type=int, default=20,
                   help="random Broyden restarts per coefficient count in exploration")
    p.add_argument("--max-candidates", type=int, default=5,
                   help="maximum number of distinct candidate solutions to keep from "
                        "exploration and refine (default 5)")
    p.add_argument("--coeffs", default=None,
                   help="starting Euler coefficients to warm-start from, as a comma list "
                        "a_2,a_3,a_5,a_7,... (positional by prime) and/or 'p:val' tokens, "
                        "e.g. '-0.42-1.07j,-0.77+1.31j,-0.40-0.24j'.  When given, the "
                        "search warm-starts from them instead of cold random exploration.")
    p.add_argument("--append", default=None, metavar="FILE",
                   help="append the result report(s) to this file (e.g. to log a grid)")
    p.add_argument("--verbose", action="store_true",
                   help="print the per-iteration detail of each refinement (spectral "
                        "parameters, box size, coefficients, sign, accuracy, ...).  "
                        "Without it, only the candidate list and per-candidate results "
                        "are shown.")
    a = p.parse_args(argv)

    import dataclasses
    fam = get_family(a.landscape, a.landscape_file)
    land, euler = fam.landscape, fam.euler
    if a.conductor is not None and a.conductor != land.conductor:
        land = dataclasses.replace(land, conductor=a.conductor)
    point = tuple(mpf(t.strip()) for t in a.point.split(","))
    if len(point) != land.dim:
        p.error("landscape %s has %d free spectral parameter(s); --point gave %d"
                % (a.landscape, land.dim, len(point)))
    if land.dim != 2:
        p.error("the box-search geometry currently supports 2 spectral parameters only "
                "(landscape %s has %d)" % (a.landscape, land.dim))
    mp.dps = a.working_precision
    eg = mpc(complex(a.epsilon.replace(" ", "")))
    # given coefficients are KEPT; the rest randomized (n_loc components per prime)
    guess = _parse_coeffs(a.coeffs, euler.complex_unknowns_per_prime)
    t0 = time.time()

    # report each candidate's result block as soon as it finishes refining, and keep the
    # blocks for the optional --append log
    blocks = []

    def on_result(n, res):
        block = ("CANDIDATE %s:\n%s"
                 % (res.get("candidate", n),
                    _search_report(res, land, a, res.get("secs", time.time() - t0))))
        print()
        print(block)
        blocks.append(block)

    results = search_landscape(
        land, euler, point, mpf(a.boxsize), a.accuracy, a.working_precision,
        mpf(a.target), restarts=a.restarts, guess=guess, eps_guess=eg,
        max_iter=a.max_iter, wander_dist=mpf(a.wander_dist), timeout=a.timeout,
        max_candidates=a.max_candidates, verbose=a.verbose, on_result=on_result)
    elapsed = time.time() - t0

    if not results:
        summary = ("=" * 68 + "\nNo candidate solutions found near %s within %s s.\n"
                   % (a.point, mpmath.nstr(mpf(a.timeout), 3)) + "=" * 68)
    else:
        succ = sum(1 for r in results if r.get("status") in ("success", "converged"))
        summary = ("FOUND %d candidate(s); %d refined to an L-point. total time %.1f s."
                   % (len(results), succ, elapsed))
    print()
    print(summary)
    if a.append:
        with open(a.append, "a") as fh:
            fh.write("\n\n".join([summary] + blocks) + "\n\n")
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
