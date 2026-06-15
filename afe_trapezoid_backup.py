#!/usr/bin/env python3
r"""
afe.py -- a smoothed approximate functional equation a la Rubinstein, Theorem 1.

This module implements the *smoothed approximate functional equation* of

    M. Rubinstein, "Computational methods and experiments in analytic number
    theory", arXiv:math/0412181, Theorem 1 (eqs. (25)-(26)),

with two deliberate departures from the paper:

  1. The auxiliary "test"/weight function is taken to be

         g(s) = s^m * exp(i*beta*s + alpha*s^2)            (m a non-negative integer)

     instead of Rubinstein's g(s) = delta^{-s}.  Because this g is not a pure
     exponential, the functions f1, f2 of Theorem 1 do NOT collapse to incomplete
     Gamma functions; they are evaluated here directly as Mellin-Barnes contour
     integrals (numerically, to a requested precision).

  2. The data of the L-function is supplied in the LMFDB Gamma_R / Gamma_C
     normalization (see lmfdb.org/knowledge/show/lfunction.functional_equation),

         Lambda(s) = N^{s/2} * prod_j Gamma_R(s + mu_j)
                              * prod_k Gamma_C(s + nu_k) * L(s),
         Lambda(s) = epsilon * conj(Lambda(1 - conj(s))),

     with  Gamma_R(s) = pi^{-s/2} Gamma(s/2),   Gamma_C(s) = 2 (2 pi)^{-s} Gamma(s).

The public entry point is `afe(...)`, which returns an `AFEResult`.  Its
`.symbolic_expression()` is a sympy expression for L(s) that is *linear in the
unknown Dirichlet coefficients*, with the real and imaginary parts

         x_n = Re b(n),    y_n = Im b(n)

kept as free real symbols and every other quantity reduced to a high-precision
number.  Internally the smoothed AFE produces Lambda(s) g(s); this is divided by
the gamma factor and the weight, L(s) = Lambda(s) g(s) / (g(s) Gfac(s)), so the
returned L(s) is independent of the choice of g.  See README.md for the
mathematics and the conversion identities.

Author: generated for an L-functions computation task.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import mpmath
from mpmath import mp, mpf, mpc


# ---------------------------------------------------------------------------
# 1.  LMFDB  ->  Rubinstein  conversion
# ---------------------------------------------------------------------------
#
# LMFDB form:
#   Lambda(s) = N^{s/2} prod_j Gamma_R(s+mu_j) prod_k Gamma_C(s+nu_k) L(s)
# with Gamma_R(s)=pi^{-s/2}Gamma(s/2), Gamma_C(s)=2(2pi)^{-s}Gamma(s).
#
# Expanding the archimedean factors,
#   Gamma_R(s+mu) = pi^{-(s+mu)/2} Gamma((s+mu)/2)        -> kappa=1/2, lambda=mu/2
#   Gamma_C(s+nu) = 2 (2pi)^{-(s+nu)} Gamma(s+nu)         -> kappa=1,   lambda=nu
# we collect the s-dependent exponential base into Q^s and the remaining constant
# into C, reaching Rubinstein's shape  Lambda(s) = C * Q^s prod_i Gamma(kappa_i s
# + lambda_i) L(s).  Concretely
#
#   Q = sqrt(N) * pi^{-d1/2} * (2pi)^{-d2},
#   C = 2^{d2} * pi^{-(sum mu_j)/2} * (2pi)^{-(sum nu_k)},
#
# where d1 = #{mu_j}, d2 = #{nu_k}.  Writing Lambda = C*Lambda_R, the LMFDB
# functional equation  Lambda(s)=eps*conj(Lambda(1-conj s))  becomes
#
#   Lambda_R(s) = omega * conj(Lambda_R(1-conj s)),   omega = eps * conj(C)/C.
#
# Theorem 1 is applied to Lambda_R and the whole identity is multiplied back by C.


@dataclass
class GammaData:
    """Rubinstein-form archimedean data plus the LMFDB conversion constants."""
    kappa: list           # kappa_i  (1/2 for each Gamma_R, 1 for each Gamma_C)
    lam: list             # lambda_i (mu_j/2 for each Gamma_R, nu_k for each Gamma_C)
    Q: mpf                # real, positive
    C: mpc                # leading constant absorbed from the conversion
    omega: mpc            # Rubinstein root number = eps * conj(C)/C
    degree: int           # d1 + 2 d2


def lmfdb_to_rubinstein(mu: Sequence, nu: Sequence, N, epsilon) -> GammaData:
    """Convert LMFDB (mu, nu, N, epsilon) data to Rubinstein (kappa, lambda, Q, C, omega)."""
    mu = [mpc(m) for m in mu]
    nu = [mpc(n) for n in nu]
    d1, d2 = len(mu), len(nu)

    kappa = [mpf(1) / 2] * d1 + [mpf(1)] * d2
    lam = [m / 2 for m in mu] + [n for n in nu]

    Q = mpmath.sqrt(mpf(N)) * mpmath.pi ** (-mpf(d1) / 2) * (2 * mpmath.pi) ** (-mpf(d2))

    sum_mu = sum(mu) if mu else mpc(0)
    sum_nu = sum(nu) if nu else mpc(0)
    C = (mpf(2) ** d2) * mpmath.pi ** (-sum_mu / 2) * (2 * mpmath.pi) ** (-sum_nu)

    omega = mpc(epsilon) * mpmath.conj(C) / C
    return GammaData(kappa=kappa, lam=lam, Q=Q, C=C, omega=omega, degree=d1 + 2 * d2)


# ---------------------------------------------------------------------------
# 2.  The weight function g(s) = s^m exp(i beta s + alpha s^2)
# ---------------------------------------------------------------------------

def make_g(m: int, alpha, beta) -> Callable:
    """Return g(s) = s^m * exp(i*beta*s + alpha*s^2) as an mpmath-valued callable."""
    m = int(m)
    a = mpc(alpha)
    b = mpc(beta)

    def g(s):
        s = mpc(s)
        poly = s ** m if m else mpc(1)
        return poly * mpmath.e ** (1j * b * s + a * s * s)

    return g


def g_admissibility(gd: GammaData, sigma, m, alpha, beta, half_strip=1.0):
    """
    Check the Theorem-1 hypothesis  |Lambda(z+s) g(z+s) z^{-1}| -> 0  as |Im z|->oo
    in the vertical strip -A <= Re z <= A (A = half_strip), for g = s^m e^{i beta s
    + alpha s^2}.

    Returns (ok: bool, message: str).

    Along Im(s+z)=T->+-oo the Gamma factors decay like exp(-(pi/2) (sum kappa_i) |T|);
    the polynomial s^m is harmless.  For g:

      * Re(alpha) > 0  -> g has Gaussian decay exp(-Re(alpha) T^2); always admissible.
      * Re(alpha) = 0  -> |g| ~ exp( -(beta_re + 2 Im(alpha) u) T ) with u=Re(s+z) in
        [sigma-A, sigma+A]; admissible iff  (pi/2) sum kappa_i > |beta_re + 2 Im(alpha) u|
        for every such u, i.e. the Gamma decay beats the linear-exponential growth.
      * Re(alpha) < 0  -> g grows like a Gaussian; never admissible.
    """
    a = mpc(alpha)
    b = mpc(beta)
    sumk = mpmath.fsum(gd.kappa)
    gamma_rate = mpmath.pi / 2 * sumk
    if mpmath.re(a) > 0:
        return True, f"Re(alpha)={mpmath.nstr(mpmath.re(a),6)}>0: Gaussian decay, admissible."
    if mpmath.re(a) < 0:
        return False, f"Re(alpha)={mpmath.nstr(mpmath.re(a),6)}<0: g grows like a Gaussian, NOT admissible."
    # Re(alpha) == 0
    br = mpmath.re(b)
    ai = mpmath.im(a)
    worst = max(abs(br + 2 * ai * (mpf(sigma) + mpf(half_strip))),
                abs(br + 2 * ai * (mpf(sigma) - mpf(half_strip))))
    ok = gamma_rate > worst
    msg = (f"Re(alpha)=0: need (pi/2)*sum(kappa)={mpmath.nstr(gamma_rate,6)} > "
           f"|beta_re + 2 Im(alpha) u|<= {mpmath.nstr(worst,6)} on the strip -> "
           f"{'admissible' if ok else 'NOT admissible'}.")
    return ok, msg


# ---------------------------------------------------------------------------
# 3.  Mellin-Barnes contour integrals f1, f2  of Theorem 1
# ---------------------------------------------------------------------------
#
#   f1(s,n)   = 1/(2 pi i) int_{(nu0)} [prod_i Gamma(kappa_i (z+s)   + lambda_i)]
#                                       z^{-1} g(s+z) (Q/n)^z dz
#   f2(1-s,n) = 1/(2 pi i) int_{(nu0)} [prod_i Gamma(kappa_i (z+1-s) + conj(lambda_i))]
#                                       z^{-1} g(s-z) (Q/n)^z dz
#
# Both are evaluated on the vertical line z = nu0 + i*tau by the trapezoidal rule,
# which is spectrally accurate for the analytic, (super-)exponentially decaying
# integrand (cf. Rubinstein sec. 2.4 / Poisson summation).  nu0 must sit to the
# right of the z^{-1} pole at 0 and of every Gamma pole.


def _min_contour(s_eff, kappa, lam):
    """Smallest admissible Re(z): right of z=0 and of all Gamma poles of the integrand."""
    bound = mpf(0)
    for k, l in zip(kappa, lam):
        # Gamma(kappa(z+s_eff)+lambda) poles at Re(z) = -Re(s_eff) - Re(lambda)/kappa - p/kappa
        rightmost = -mpmath.re(s_eff) - mpmath.re(l) / k
        if rightmost > bound:
            bound = rightmost
    return bound


def _line_integral(Ftau, tol, h0=mpf(1) / 2, max_halvings=24, max_width_doublings=40):
    """
    Compute int_{-inf}^{inf} Ftau(tau) d tau by the trapezoidal rule with adaptive
    half-width W and step h, returning (value, abs_error_estimate).

    The integrand is analytic and (super-)exponentially decaying, so the
    trapezoidal rule is spectrally accurate (cf. Rubinstein sec. 2.4 / Poisson
    summation).  Step refinement reuses previously computed nodes (halving h only
    needs the new midpoints) and continues until either the requested relative
    tolerance `tol` is met or the increments stop shrinking (the working-precision
    round-off floor is reached).  The returned error estimate is the size of the
    last refinement step, floored at the round-off level -- i.e. the *actual*
    accuracy achieved, not a requested target.
    """
    roundoff = mpf(10) ** (-(mp.dps - 1))

    # scale of the integrand near the centre
    scale = mpmath.fsum(abs(Ftau(mpf(t))) for t in (0, 1, -1, 2, -2)) / 5
    if scale == 0:
        scale = mpf(1)

    # half-width W chosen so the tails are below the round-off floor
    W = mpf(6)
    tail_floor = roundoff * scale
    for _ in range(max_width_doublings):
        if abs(Ftau(W)) + abs(Ftau(-W)) < tail_floor or W > mpf(10) ** 4:
            break
        W *= 2

    # initial grid: nodes k*h for k = -K..K
    h = h0
    K = int(mpmath.ceil(W / h))
    node_sum = mpmath.fsum(Ftau(k * h) for k in range(-K, K + 1))
    val = h * node_sum
    change = abs(val)
    prev_change = None
    for _ in range(max_halvings):
        # halve the step by inserting the midpoints (k + 1/2) * h
        mids = mpmath.fsum(Ftau((k + mpf(1) / 2) * h) for k in range(-K - 1, K + 1))
        node_sum += mids
        h /= 2
        K = 2 * K + 1
        new = h * node_sum
        change = abs(new - val)
        val = new
        if change < tol * max(abs(val), scale):
            break
        # stop once the increments quit shrinking: round-off floor reached
        if prev_change is not None and change >= prev_change / 2:
            break
        prev_change = change
    err = max(change, roundoff * max(abs(val), scale))
    return val, err


def f1(s, n, gd: GammaData, g: Callable, tol, contour_nu=None):
    """f1(s,n) of Theorem 1 (the term multiplying b(n)/n^s); returns (value, err)."""
    s = mpc(s)
    Qn = gd.Q / mpf(n)
    logQn = mpmath.log(Qn)
    nu0 = mpf(contour_nu) if contour_nu is not None else _min_contour(s, gd.kappa, gd.lam) + mpf(1) / 2

    def Ftau(tau):
        z = mpc(nu0, tau)
        prod = mpc(1)
        for k, l in zip(gd.kappa, gd.lam):
            prod *= mpmath.gamma(k * (z + s) + l)
        return prod / z * g(s + z) * mpmath.e ** (z * logQn)

    val, err = _line_integral(Ftau, tol)
    twopi = 2 * mpmath.pi
    return val / twopi, err / twopi


def f2(s, n, gd: GammaData, g: Callable, tol, contour_nu=None):
    """f2(1-s,n) of Theorem 1 (the term multiplying conj(b(n))/n^{1-s}).

    The argument `s` is the same s as in Lambda(s); internally the first Gamma slot
    carries (z + 1 - s) and the conjugated spectral parameters conj(lambda_i), while
    the weight is evaluated at g(s - z)."""
    s = mpc(s)
    one_minus_s = 1 - s
    Qn = gd.Q / mpf(n)
    logQn = mpmath.log(Qn)
    lam_conj = [mpmath.conj(l) for l in gd.lam]
    nu0 = mpf(contour_nu) if contour_nu is not None else _min_contour(one_minus_s, gd.kappa, lam_conj) + mpf(1) / 2

    def Ftau(tau):
        z = mpc(nu0, tau)
        prod = mpc(1)
        for k, l in zip(gd.kappa, lam_conj):
            prod *= mpmath.gamma(k * (z + one_minus_s) + l)
        return prod / z * g(s - z) * mpmath.e ** (z * logQn)

    val, err = _line_integral(Ftau, tol)
    twopi = 2 * mpmath.pi
    return val / twopi, err / twopi


# ---------------------------------------------------------------------------
# 4.  The result object
# ---------------------------------------------------------------------------

def _sig_digits(value, abserr, cap):
    """Number of significant digits of `value` that are actually correct, given an
    absolute error estimate `abserr` (capped at `cap`)."""
    av = abs(mpc(value))
    ae = abs(abserr)
    if av == 0:
        return 1
    if ae <= 0:
        return int(cap)
    d = int(mpmath.floor(mpmath.log10(av) - mpmath.log10(ae)))
    return max(1, min(d, int(cap)))


@dataclass
class AFEResult:
    """
    Result of the smoothed approximate functional equation, solved for L(s).

    The identity produced is

        L(s)  =  pole_term  +  sum_{n=1}^{M} [ cx[n] * x_n  +  cy[n] * y_n ]

    where x_n = Re b(n), y_n = Im b(n) are the unknown coefficient parts and
    M = num_terms.  These come from the smoothed AFE for Lambda(s) g(s) divided by
    D = g(s) * Gfac(s), where Gfac(s) = C Q^s prod_i Gamma(kappa_i s + lambda_i) is
    the complete gamma factor (Lambda = Gfac * L).  Concretely (1-indexed)
        A_n = C Q^s n^{-s}    f1(s,n)   / D     (coefficient of b(n)),
        B_n = C omega Q^{1-s} n^{s-1} f2(1-s,n) / D (coefficient of conj(b(n))),
        cx[n] = A_n + B_n,
        cy[n] = i (A_n - B_n).
    The weight g(s) cancels in D, so L(s) does not depend on the choice of g.
    All of cx, cy, pole_term are high-precision complex numbers (mpc).
    """
    s: mpc
    pole_term: mpc
    cx: list                      # cx[n-1] multiplies x_n = Re b(n)
    cy: list                      # cy[n-1] multiplies y_n = Im b(n)
    A: list                       # A[n-1], the b(n)/n^s coefficient
    B: list                       # B[n-1], the conj(b(n))/n^{1-s} coefficient
    num_terms: int
    accuracy: int
    working_precision: int
    gd: GammaData
    info: dict = field(default_factory=dict)
    # absolute-error estimates (actual achieved accuracy) for each quantity
    cx_err: list = field(default_factory=list)
    cy_err: list = field(default_factory=list)
    A_err: list = field(default_factory=list)
    B_err: list = field(default_factory=list)
    pole_err: object = mpf(0)

    # -- evaluate at given numeric coefficients ----------------------------
    def evaluate(self, b):
        """Plug in numeric coefficients b = [b(1), b(2), ...] and return L(s)."""
        return self.evaluate_with_error(b)[0]

    def evaluate_with_error(self, b):
        """Like evaluate(), but also return an absolute-error estimate for L(s),
        combining the per-coefficient errors with a truncation-tail estimate."""
        total = self.pole_term
        err = abs(self.pole_err)
        last = mpf(0)
        for n in range(1, self.num_terms + 1):
            bn = mpc(b[n - 1])
            term = self.A[n - 1] * bn + self.B[n - 1] * mpmath.conj(bn)
            total += term
            err += abs(bn) * (self.A_err[n - 1] + self.B_err[n - 1])
            last = abs(term)
        err += last      # truncation tail ~ size of the last retained term
        return total, err

    # -- known-coefficient handling ---------------------------------------
    @staticmethod
    def _normalize_known(known):
        """Normalize `known` to a dict {n: value} (1-based), dropping None entries."""
        if known is None:
            return {}
        if isinstance(known, dict):
            return {int(n): v for n, v in known.items() if v is not None}
        return {i + 1: v for i, v in enumerate(known) if v is not None}

    # -- symbolic expression ----------------------------------------------
    def symbolic_expression(self, known=None, digits: Optional[int] = None,
                            prefix_x="x", prefix_y="y"):
        """
        Build a sympy expression for L(s), linear in the real symbols
        x_n = Re b(n) and y_n = Im b(n).  `digits` controls how many significant
        decimal digits of each coefficient are emitted (default: self.accuracy).

        If `known` is given (a dict {n: b(n)} or a sequence [b(1), b(2), ...], with
        None to leave an entry symbolic), those coefficients are substituted with
        their numeric values and the remaining ones are left symbolic.  Indices
        beyond num_terms are ignored (their contribution is below the truncation
        level).  Supplying every coefficient yields a single (complex) number.

        By default each number is emitted with as many significant figures as are
        actually correct (estimated from the computation's error, capped at the
        working precision).  Pass an explicit `digits` to force a fixed count.
        """
        import sympy
        cap = self.working_precision
        kmap = self._normalize_known(known)

        def cnum(z, err):
            z = mpc(z)
            d = int(digits) if digits is not None else _sig_digits(z, err, cap)
            re = sympy.Float(mpmath.nstr(mpmath.re(z), d, strip_zeros=False), d)
            im = sympy.Float(mpmath.nstr(mpmath.im(z), d, strip_zeros=False), d)
            return re + sympy.I * im

        expr = cnum(self.pole_term, self.pole_err)
        for n in range(1, self.num_terms + 1):
            if n in kmap:
                v = mpc(kmap[n])     # substitute b(n): contributes cx_n*Re v + cy_n*Im v
                contrib = self.cx[n - 1] * mpmath.re(v) + self.cy[n - 1] * mpmath.im(v)
                cerr = self.cx_err[n - 1] * abs(mpmath.re(v)) + self.cy_err[n - 1] * abs(mpmath.im(v))
                expr += cnum(contrib, cerr)
            else:
                xn = sympy.Symbol(f"{prefix_x}{n}", real=True)
                yn = sympy.Symbol(f"{prefix_y}{n}", real=True)
                expr += cnum(self.cx[n - 1], self.cx_err[n - 1]) * xn \
                    + cnum(self.cy[n - 1], self.cy_err[n - 1]) * yn
        return expr

    def substitute(self, known, digits: Optional[int] = None, prefix_x="x", prefix_y="y"):
        """Convenience alias: symbolic L(s) with the `known` coefficients substituted."""
        return self.symbolic_expression(known=known, digits=digits,
                                        prefix_x=prefix_x, prefix_y=prefix_y)

    def __str__(self):
        lines = [
            f"AFEResult for L({mpmath.nstr(self.s, 12).strip('()')})",
            f"  degree {self.gd.degree},  Q = {mpmath.nstr(self.gd.Q, 12)},  "
            f"omega = {mpmath.nstr(self.gd.omega, 12)}",
            f"  terms used: {self.num_terms},  accuracy: {self.accuracy} digits,  "
            f"working precision: {self.working_precision} dps",
            f"  pole term = {mpmath.nstr(self.pole_term, _sig_digits(self.pole_term, self.pole_err, self.working_precision))}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5.  The main driver
# ---------------------------------------------------------------------------

def afe(
    s,
    mu: Sequence = (),
    nu: Sequence = (),
    N=1,
    epsilon=1,
    poles: Sequence = (),          # iterable of (s_k, r_k): poles & residues of Lambda (LMFDB)
    g_m: int = 0,
    g_alpha=0,
    g_beta=0,
    accuracy: int = 20,
    working_precision: Optional[int] = None,
    num_terms: Optional[int] = None,
    coeff_growth: float = 0.0,      # assume |b(n)| = O(n^coeff_growth) for truncation
    contour_nu=None,
    patience: int = 4,
    max_terms: int = 100000,
) -> AFEResult:
    """
    Smoothed approximate functional equation (Rubinstein Thm 1) with weight
    g(s) = s^{g_m} exp(i*g_beta*s + g_alpha*s^2), L-function given in LMFDB form.

    Parameters
    ----------
    s                 : point of evaluation (complex).
    mu, nu            : spectral parameters of the Gamma_R / Gamma_C factors.
    N                 : conductor.
    epsilon           : LMFDB root number (sign of the functional equation).
    poles             : poles (s_k) and residues (r_k) of the *completed* Lambda(s).
    g_m, g_alpha, g_beta : parameters of the weight g.
    accuracy          : desired number of correct decimal digits of the output.
    working_precision : mpmath dps used internally (default: accuracy + guard).
    num_terms         : if given, force exactly this many Dirichlet terms.
    coeff_growth      : exponent r in the assumed bound |b(n)| = O(n^r), used only
                        to decide where to truncate.
    contour_nu        : override the real part of the Mellin-Barnes contour.
    patience          : stop after this many consecutive negligible terms.

    Returns
    -------
    AFEResult
    """
    if working_precision is None:
        gd_degree_guess = len(mu) + 2 * len(nu)
        working_precision = int(accuracy) + 15 + 3 * max(1, gd_degree_guess)

    old_dps = mp.dps
    mp.dps = working_precision
    try:
        s = mpc(s)
        gd = lmfdb_to_rubinstein(mu, nu, N, epsilon)
        g = make_g(g_m, g_alpha, g_beta)

        ok, msg = g_admissibility(gd, mpmath.re(s), g_m, g_alpha, g_beta)

        tol = mpmath.mpf(10) ** (-int(accuracy))                  # truncation target
        int_tol = mpmath.mpf(10) ** (-(working_precision - 3))    # contour-integral target
        roundoff = mpmath.mpf(10) ** (-(working_precision - 1))

        # pole / residue term:  sum_k r_k g(s_k) / (s - s_k)  (closed form)
        pole_term = mpc(0)
        for (sk, rk) in poles:
            sk = mpc(sk)
            pole_term += mpc(rk) * g(sk) / (s - sk)

        Qs = mpmath.e ** (s * mpmath.log(gd.Q))             # Q^s
        Q1s = mpmath.e ** ((1 - s) * mpmath.log(gd.Q))      # Q^{1-s}

        A_list, B_list, A_err, B_err = [], [], [], []
        consecutive_small = 0
        n = 0
        while True:
            n += 1
            # contour integrals are computed to (near) the working precision; the
            # returned error is the accuracy actually achieved, not the target
            f1v, f1e = f1(s, n, gd, g, int_tol, contour_nu=contour_nu)
            f2v, f2e = f2(s, n, gd, g, int_tol, contour_nu=contour_nu)
            prefA = gd.C * Qs * mpmath.e ** (-s * mpmath.log(n))
            prefB = gd.C * gd.omega * Q1s * mpmath.e ** (-(1 - s) * mpmath.log(n))
            A_n = prefA * f1v
            B_n = prefB * f2v
            A_list.append(A_n)
            B_list.append(B_n)
            A_err.append(max(abs(prefA) * f1e, roundoff * abs(A_n)))
            B_err.append(max(abs(prefB) * f2e, roundoff * abs(B_n)))

            if num_terms is not None:
                if n >= num_terms:
                    break
                continue

            # Truncate when the term is negligible.  The extra factor `tail_margin`
            # guards against a slowly-decaying geometric tail (relevant for a
            # Gaussian-damped g, whose terms fall off only like exp(-c n));
            # for the classical incomplete-Gamma rate it costs nothing.
            tail_margin = mpf(10) ** (-2)
            weight = mpf(n) ** mpf(coeff_growth)
            size = (abs(A_n) + abs(B_n)) * weight
            if size < tol * tail_margin:
                consecutive_small += 1
                if consecutive_small >= patience and n >= 1:
                    break
            else:
                consecutive_small = 0
            if n >= max_terms:
                break

        # Convert the smoothed AFE for Lambda(s) g(s) into L(s) itself.  Since
        #   Lambda(s) = Gfac(s) L(s),   Gfac(s) = C Q^s prod_i Gamma(kappa_i s + lambda_i)
        # (the complete N^{s/2} prod Gamma_R prod Gamma_C factor), we have
        #   L(s) = Lambda(s) g(s) / ( g(s) Gfac(s) ).
        # The explicit weight g(s) cancels, so the returned L(s) is independent of
        # the choice of g (a useful internal check); g still governs convergence.
        gfac = gd.C * Qs * mpmath.fprod([mpmath.gamma(k * s + l)
                                         for k, l in zip(gd.kappa, gd.lam)])
        gs = g(s)
        D = gs * gfac
        if D == 0:
            raise ValueError(
                f"cannot recover L(s): g(s)*gamma_factor = 0 at s={s} (g(s)={gs}). "
                f"Choose a different s or weight g (e.g. avoid s=0 when g(s)=s^m).")
        absD = abs(D)
        pole_term = pole_term / D
        A_list = [a / D for a in A_list]
        B_list = [b / D for b in B_list]
        A_err = [e / absD for e in A_err]
        B_err = [e / absD for e in B_err]
        cx = [A_list[i] + B_list[i] for i in range(len(A_list))]
        cy = [1j * (A_list[i] - B_list[i]) for i in range(len(A_list))]
        cx_err = [A_err[i] + B_err[i] for i in range(len(A_list))]
        cy_err = [A_err[i] + B_err[i] for i in range(len(A_list))]
        pole_err = roundoff * abs(pole_term)

        result = AFEResult(
            s=s, pole_term=pole_term, cx=cx, cy=cy, A=A_list, B=B_list,
            num_terms=len(cx), accuracy=int(accuracy), working_precision=working_precision,
            gd=gd, info={"g_admissible": ok, "g_admissibility_msg": msg,
                         "g": (g_m, mpc(g_alpha), mpc(g_beta)),
                         "gamma_factor": gfac, "divisor": D},
            cx_err=cx_err, cy_err=cy_err, A_err=A_err, B_err=B_err, pole_err=pole_err,
        )
        return result
    finally:
        mp.dps = old_dps


def afe_substitute(s, coefficients, accuracy: int = 20,
                   working_precision: Optional[int] = None,
                   digits: Optional[int] = None, **afe_kwargs):
    """
    Smoothed approximate functional equation for L(s) with a *partial* set of
    Dirichlet coefficients already substituted.

    This is the companion to `afe`: it computes the same L(s) expansion and then
    plugs in the coefficients the caller knows, returning the L-function with
    those values in place and the unknown coefficients still symbolic.

    Parameters
    ----------
    s             : point of evaluation (complex).
    coefficients  : the known coefficients, as a dict {n: b(n)} or a sequence
                    [b(1), b(2), ...]; use None in a sequence to leave that entry
                    symbolic.  Indices beyond the truncation length are ignored.
    accuracy          : desired number of correct decimal digits of the output.
    working_precision : the internal mpmath precision (mp.dps); default auto
                        (accuracy + guard digits).
    digits        : significant digits emitted per numeric coefficient.
    **afe_kwargs  : everything else (mu, nu, N, epsilon, poles, g_m, g_alpha,
                    g_beta, num_terms, ...) is passed straight through to `afe`.

    Returns
    -------
    (expr, result) : `expr` is a sympy expression for L(s) — numeric in the
    supplied coefficients, symbolic (in x_n = Re b(n), y_n = Im b(n)) in the
    rest; if every coefficient is supplied it is a single complex number.
    `result` is the underlying AFEResult.

    Example
    -------
    >>> # L(s) of zeta with b(1)=1, b(2)=1 fixed, the rest left symbolic:
    >>> expr, res = afe_substitute(2+3j, {1: 1, 2: 1}, mu=[0], N=1, epsilon=1,
    ...                            poles=[(1, 1), (0, -1)], accuracy=12)
    """
    res = afe(s, accuracy=accuracy, working_precision=working_precision, **afe_kwargs)
    expr = res.substitute(coefficients, digits=digits)
    return expr, res


# ---------------------------------------------------------------------------
# 6.  Self-tests / validation against known L-functions
# ---------------------------------------------------------------------------

def _zeta_lambda(s):
    """Completed Riemann zeta Lambda(s) = pi^{-s/2} Gamma(s/2) zeta(s) (LMFDB norm)."""
    return mpmath.pi ** (-s / 2) * mpmath.gamma(s / 2) * mpmath.zeta(s)


def _dirichlet_char_mod5_order4():
    """A primitive complex (odd) character chi mod 5 of order 4.  Returns chi(n)."""
    # 2 is a primitive root mod 5: 2^0=1,2^1=2,2^2=4,2^3=3.  Set chi(2)=i.
    table = {1: mpc(1), 2: mpc(0, 1), 4: mpc(-1), 3: mpc(0, -1), 0: mpc(0)}

    def chi(n):
        return table[n % 5]

    return chi, 5


def _dirichlet_L(chi, q, s):
    """L(s,chi) via Hurwitz zeta:  q^{-s} sum_{a=1}^{q} chi(a) zeta(s, a/q)."""
    return mpf(q) ** (-s) * mpmath.fsum(chi(a) * mpmath.zeta(s, mpf(a) / q) for a in range(1, q + 1))


def _gauss_sum(chi, q):
    return mpmath.fsum(chi(a) * mpmath.e ** (2j * mpmath.pi * a / q) for a in range(1, q + 1))


# --- Elliptic curve 11.a  (LMFDB L-function 2-11-1.1-c1-0-0) ----------------
# Data taken from the LMFDB (www.lmfdb.org).  In LMFDB analytic normalization
# this is a degree-2, self-dual L-function with conductor N = 11, one Gamma_C
# factor with nu = 1/2, root number epsilon = +1, analytic rank 0, and central
# value L(1/2) = 0.2538418608559107 (= L(E,1) of the curve 11.a).  The Frobenius
# traces a_p below are read from the LMFDB Euler factors 1 - a_p X + p X^2.
EC_11A = {
    "N": 11, "nu": [mpf(1) / 2], "mu": [], "epsilon": 1,
    "central_value": "0.2538418608559107",
    "ap": {2: -2, 3: -1, 5: 1, 7: -2, 13: 4, 17: -2, 19: 0, 23: -1, 29: 0,
           31: 7, 37: 3, 41: -8, 43: -6, 47: 8, 53: -6, 59: 5, 61: 12, 67: -7,
           71: -3, 73: 4, 79: -10, 83: -6, 89: 15, 97: -7, 101: 2, 103: -16,
           107: 18, 109: 10, 113: 9},
    "bad_ap": {11: 1},          # split multiplicative reduction at 11: a_11 = 1
}


def _ap_power(p, e, ap, bad):
    """a_{p^e} from a_p, via the Hecke recurrence (good p) or a_{p^e}=a_p^e (bad p)."""
    if p in bad:
        return bad[p] ** e
    if e == 0:
        return 1
    if e == 1:
        return ap[p]
    prev2, prev1 = 1, ap[p]
    for _ in range(2, e + 1):
        prev2, prev1 = prev1, ap[p] * prev1 - p * prev2
    return prev1


def _anlist_from_ap(ap, bad, M):
    """Arithmetic coefficients a_1..a_M from the a_p, by Hecke multiplicativity."""
    spf = list(range(M + 1))          # smallest prime factor
    i = 2
    while i * i <= M:
        if spf[i] == i:
            for j in range(i * i, M + 1, i):
                if spf[j] == j:
                    spf[j] = i
        i += 1
    a = [0] * (M + 1)
    if M >= 1:
        a[1] = 1
    for n in range(2, M + 1):
        m, val = n, 1
        while m > 1:
            p = spf[m]
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            val *= _ap_power(p, e, ap, bad)
        a[n] = val
    return a


def elliptic_curve_11a_bcoeffs(M):
    """Analytic Dirichlet coefficients b(n) = a_n / sqrt(n) of the curve 11.a, n=1..M."""
    a = _anlist_from_ap(EC_11A["ap"], EC_11A["bad_ap"], M)
    return [mpc(a[n]) / mpmath.sqrt(n) for n in range(1, M + 1)]


def selftest(accuracy=20, verbose=True):
    """
    Validate the implementation by checking that the right-hand side of the AFE
    reproduces the directly-computed value of Lambda(s) g(s).

    The rigorous, fast checks use the classical weight g = 1, where the
    convergence in n is super-exponential (incomplete-Gamma rate); they also
    verify f1 against the closed-form incomplete Gamma function and exercise the
    complex-coefficient x_n/y_n split.  A final check demonstrates the requested
    Gaussian weight g(s) = s^2 exp(i*beta*s + alpha*s^2) (whose convergence in n
    is much slower), at a modest accuracy to keep it quick.
    """
    results = []   # (name, num_terms, rel_err, pass_threshold)

    def threshold_for(acc):
        return mpmath.mpf(10) ** (-(int(acc) - 3))

    wp_fast = accuracy + 6        # modest working precision to keep the selftest quick

    old = mp.dps
    mp.dps = accuracy + 25
    try:
        # ===== A. g = 1, Riemann zeta, s = 2 + 3i ==========================
        # A1: f1(s,n) must equal the upper incomplete Gamma Gamma(s/2, pi n^2).
        s = mpc(2, 3)
        g1 = make_g(0, 0, 0)
        gd_z = lmfdb_to_rubinstein([0], [], 1, 1)
        tol = mpmath.mpf(10) ** (-accuracy)
        # Compare against the closed form.  The meaningful metric is the ABSOLUTE
        # error (terms are summed): once |Gamma(s/2,pi n^2)| drops below the
        # integrator's absolute floor ~tol, its relative error is irrelevant.
        rel1 = None
        abs_max = mpf(0)
        for n in (1, 2, 3):
            got, _ = f1(s, n, gd_z, g1, tol)
            want = mpmath.gammainc(s / 2, mpmath.pi * n * n)   # Gamma(s/2, pi n^2)
            abs_max = max(abs_max, abs(got - want))
            if n == 1:
                rel1 = abs(got - want) / abs(want)
        results.append(("f1 vs incomplete Gamma (g=1), abs", 3, abs_max, tol * 100))
        if verbose:
            print(f"[zeta]   f1(s,n) vs Gamma(s/2,pi n^2), n=1..3  "
                  f"max.abs.err={mpmath.nstr(abs_max,3)}  rel.err(n=1)={mpmath.nstr(rel1,3)}")

        # A2: full AFE reproduces L(s) = zeta(s).
        res = afe(s, mu=[0], nu=[], N=1, epsilon=1, poles=[(1, 1), (0, -1)],
                  g_m=0, g_alpha=0, g_beta=0, accuracy=accuracy, working_precision=wp_fast)
        rhs = res.evaluate([mpc(1)] * res.num_terms)
        lhs = mpmath.zeta(s)
        err = abs(lhs - rhs) / abs(lhs)
        results.append(("zeta: L(s) at s=2+3i (g=1)", res.num_terms, err, threshold_for(accuracy)))
        if verbose:
            print(f"[zeta]   L(s) at s={mpmath.nstr(s,6)}  terms={res.num_terms}  rel.err={mpmath.nstr(err,3)}")
            print(f"         zeta(s)   ={mpmath.nstr(lhs, 14)}")
            print(f"         AFE L(s)  ={mpmath.nstr(rhs, 14)}")

        # ===== B. g = 1, Dirichlet L, complex primitive char mod 5 (odd) ====
        chi, q = _dirichlet_char_mod5_order4()
        s = mpc(mpf("0.7"), mpf("1.1"))
        tau = _gauss_sum(chi, q)
        epsilon = tau / (1j * mpmath.sqrt(q))             # root number (odd char, a=1)
        res = afe(s, mu=[1], nu=[], N=q, epsilon=epsilon, poles=[],
                  g_m=0, g_alpha=0, g_beta=0, accuracy=accuracy, working_precision=wp_fast)
        rhs = res.evaluate([chi(n) for n in range(1, res.num_terms + 1)])
        lhs = _dirichlet_L(chi, q, s)
        err = abs(lhs - rhs) / abs(lhs)
        results.append(("Dirichlet L(s) mod 5, complex chi (g=1)", res.num_terms, err, threshold_for(accuracy)))
        if verbose:
            print(f"[DirL]   L(s) at s={mpmath.nstr(s,6)}  q=5 odd order-4 chi  "
                  f"terms={res.num_terms}  rel.err={mpmath.nstr(err,3)}")
            print(f"         |epsilon|={mpmath.nstr(abs(epsilon),8)} (should be 1); "
                  f"complex coeffs exercise the x_n/y_n split")

        # ===== C. requested Gaussian weight, zeta (g-independence of L(s)) ===
        # The Gaussian g converges slowly in n, so cap the term count and use a
        # matching threshold; the point is that L(s) comes out the same as g=1.
        s = mpc(2, 3)
        g_m, g_alpha, g_beta = 2, mpf("0.2"), mpf("0.5")
        res = afe(s, mu=[0], nu=[], N=1, epsilon=1, poles=[(1, 1), (0, -1)],
                  g_m=g_m, g_alpha=g_alpha, g_beta=g_beta, accuracy=accuracy,
                  working_precision=20, num_terms=70)
        rhs = res.evaluate([mpc(1)] * res.num_terms)
        lhs = mpmath.zeta(s)                       # L(s) must not depend on g
        errC = abs(lhs - rhs) / abs(lhs)
        results.append(("zeta: L(s), Gaussian g (g-independence)",
                        res.num_terms, errC, mpf(10) ** (-5)))
        if verbose:
            print(f"[zeta]   L(s) with g=s^2 exp(0.5 i s + 0.2 s^2)  "
                  f"terms={res.num_terms}  rel.err={mpmath.nstr(errC,3)} (same L(s) as g=1)")
            print(f"         admissible: {res.info['g_admissibility_msg']}")

        # ===== D. Gamma_C path vs Gamma_R duplication (degree-2 consistency) =
        # Gamma_C(s+nu) = Gamma_R(s+nu) Gamma_R(s+nu+1) (Legendre duplication), so
        # the per-term coefficients must agree whether the data is supplied as one
        # Gamma_C(nu) or as two Gamma_R(nu), Gamma_R(nu+1).
        sD = mpc(mpf("0.6"), 2)
        common = dict(N=11, epsilon=1, poles=[], g_m=0, g_alpha=0, g_beta=0,
                      accuracy=accuracy, working_precision=wp_fast, num_terms=8)
        rC = afe(sD, mu=[], nu=[2], **common)
        rR = afe(sD, mu=[2, 3], nu=[], **common)
        dmax = mpf(0)
        for n in range(rC.num_terms):
            dmax = max(dmax,
                       abs(rC.A[n] - rR.A[n]) / (abs(rR.A[n]) + tol),
                       abs(rC.B[n] - rR.B[n]) / (abs(rR.B[n]) + tol))
        results.append(("Gamma_C path == Gamma_R duplication", rC.num_terms, dmax,
                        threshold_for(accuracy)))
        if verbose:
            print(f"[degC]   Gamma_C(nu=2) vs Gamma_R(2),Gamma_R(3) at s={mpmath.nstr(sD,5)}  "
                  f"max.rel.diff={mpmath.nstr(dmax,3)}")

        # ===== E. Elliptic curve 11.a (LMFDB data): central value ============
        # Validate against the LMFDB central value L(1/2)=0.2538418608559107 using
        # the LMFDB coefficients/gamma factors/sign.  Degree 2, one Gamma_C(1/2).
        acc_ec = min(accuracy, 13)
        bco = elliptic_curve_11a_bcoeffs(120)
        rE = afe(mpf(1) / 2, mu=EC_11A["mu"], nu=EC_11A["nu"], N=EC_11A["N"],
                 epsilon=EC_11A["epsilon"], poles=[], g_m=0, g_alpha=0, g_beta=0,
                 accuracy=acc_ec, working_precision=20)
        L_half = rE.evaluate(bco[:rE.num_terms])               # L(1/2) directly
        want = mpf(EC_11A["central_value"])
        errE = abs(L_half - want) / want
        results.append(("Elliptic curve 11.a, L(1/2) vs LMFDB", rE.num_terms, errE,
                        threshold_for(acc_ec)))
        if verbose:
            print(f"[ec11a]  L(1/2) via AFE = {mpmath.nstr(mpmath.re(L_half), 16)}  "
                  f"(Im={mpmath.nstr(mpmath.im(L_half),2)})")
            print(f"         LMFDB L(1/2)   = {mpmath.nstr(want, 16)}  "
                  f"terms={rE.num_terms}  rel.err={mpmath.nstr(errE,3)}")
    finally:
        mp.dps = old

    ok = all(err < thr for (_, _, err, thr) in results)
    print()
    print("SELFTEST", "PASS" if ok else "FAIL")
    for name, nt, err, thr in results:
        print(f"   {'ok ' if err < thr else 'BAD'} {name:40s} "
              f"terms={nt:4d}  rel.err={mpmath.nstr(err,3):10s}  (<{mpmath.nstr(thr,2)})")
    return ok


# ---------------------------------------------------------------------------
# 7.  Command-line interface
# ---------------------------------------------------------------------------

def _parse_complex_list(text):
    """Parse '0,1,0.5+1j' into a list of complex numbers (empty string -> [])."""
    text = (text or "").strip()
    if not text:
        return []
    return [complex(tok.replace(" ", "")) for tok in text.split(",")]


def _parse_complex(text):
    return complex(str(text).replace(" ", ""))


def _parse_poles(text):
    """Parse 's1:r1;s2:r2' (e.g. '1:1;0:-1') into [(s1,r1),(s2,r2)]."""
    text = (text or "").strip()
    if not text:
        return []
    out = []
    for piece in text.split(";"):
        sk, rk = piece.split(":")
        out.append((complex(sk.replace(" ", "")), complex(rk.replace(" ", ""))))
    return out


def _parse_known(text):
    """Parse known coefficients: 'b1,b2,...' positionally and/or 'n:val' tokens.

    Examples: '1,1,1' -> {1:1,2:1,3:1};  '1:1,3:-2' -> {1:1, 3:-2}."""
    text = (text or "").strip()
    if not text:
        return {}
    out, pos = {}, 0
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            n, v = tok.split(":")
            out[int(n)] = complex(v.replace(" ", ""))
        else:
            pos += 1
            out[pos] = complex(tok.replace(" ", ""))
    return out


def _add_common_args(p):
    p.add_argument("--mu", default="", help="comma list of Gamma_R shifts mu_j, e.g. '0' or '0,1'")
    p.add_argument("--nu", default="", help="comma list of Gamma_C shifts nu_k, e.g. '0.5'")
    p.add_argument("--N", default="1", help="conductor N (integer)")
    p.add_argument("--epsilon", default="1", help="LMFDB root number, e.g. '1' or '0.6+0.8j'")
    p.add_argument("--poles", default="", help="poles;residues of Lambda, e.g. '1:1;0:-1'")
    p.add_argument("--m", type=int, default=0, help="power m in g(s)=s^m exp(i*beta*s+alpha*s^2)")
    p.add_argument("--alpha", default="0", help="alpha in g (need Re(alpha)>=0), e.g. '0.2' or '0.1j'")
    p.add_argument("--beta", default="0", help="beta in g, e.g. '0.5'")
    p.add_argument("--accuracy", type=int, default=20, help="desired digits of accuracy")
    p.add_argument("--working-precision", type=int, default=None, help="mpmath dps (default: auto)")
    p.add_argument("--num-terms", type=int, default=None, help="force a fixed number of Dirichlet terms")
    p.add_argument("--coeff-growth", type=float, default=0.0, help="r in assumed |b(n)|=O(n^r)")
    p.add_argument("--contour-nu", default=None, help="override Re(z) of the Mellin-Barnes contour")


def _build_from_args(args):
    return afe(
        s=_parse_complex(args.s),
        mu=_parse_complex_list(args.mu),
        nu=_parse_complex_list(args.nu),
        N=int(args.N),
        epsilon=_parse_complex(args.epsilon),
        poles=_parse_poles(args.poles),
        g_m=args.m, g_alpha=_parse_complex(args.alpha), g_beta=_parse_complex(args.beta),
        accuracy=args.accuracy,
        working_precision=args.working_precision,
        num_terms=args.num_terms,
        coeff_growth=args.coeff_growth,
        contour_nu=(None if args.contour_nu is None else _parse_complex(args.contour_nu)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="afe.py",
        description="Smoothed approximate functional equation (Rubinstein Thm 1) "
                    "with weight g(s)=s^m exp(i*beta*s+alpha*s^2), LMFDB Gamma_R/Gamma_C data.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("expr", help="print the symbolic expression for L(s)")
    pe.add_argument("--s", required=True, help="evaluation point, e.g. '0.5+14.1j'")
    _add_common_args(pe)
    pe.add_argument("--digits", type=int, default=None, help="digits shown per coefficient")
    pe.add_argument("--known", default="",
                    help="known coefficients to substitute: 'b1,b2,...' positionally "
                         "and/or 'n:val' tokens, e.g. '1,1,1' or '1:1,3:-2'")
    pe.add_argument("--table", action="store_true", help="also print the raw coefficient table")

    pv = sub.add_parser("eval", help="evaluate L(s) at supplied numeric coefficients")
    pv.add_argument("--s", required=True)
    _add_common_args(pv)
    pv.add_argument("--coeffs", required=True,
                    help="comma list b(1),b(2),... e.g. '1,1,1,1' (defines num-terms)")

    ps = sub.add_parser("selftest", help="validate against zeta and Dirichlet L-functions")
    ps.add_argument("--accuracy", type=int, default=20)

    args = parser.parse_args(argv)

    if args.cmd == "selftest":
        ok = selftest(accuracy=args.accuracy)
        raise SystemExit(0 if ok else 1)

    if args.cmd == "expr":
        res = _build_from_args(args)
        print("# " + str(res).replace("\n", "\n# "))
        if not res.info["g_admissible"]:
            print("# WARNING: " + res.info["g_admissibility_msg"])
        known = _parse_known(args.known)
        expr = res.symbolic_expression(known=known, digits=args.digits)
        sval = mpmath.nstr(res.s, 12).strip("()")
        print()
        if known:
            ks = ",".join(str(n) for n in sorted(known))
            print(f"L({sval}) =   (coefficients b(n) substituted for n in {{{ks}}})")
        else:
            print(f"L({sval}) =")
        print(expr)
        if args.table:
            print()
            print("# n   A_n (coeff of b(n))                B_n (coeff of conj(b(n)))")
            wp = res.working_precision
            for n in range(1, res.num_terms + 1):
                da = _sig_digits(res.A[n - 1], res.A_err[n - 1], wp)
                db = _sig_digits(res.B[n - 1], res.B_err[n - 1], wp)
                print(f"# {n:3d}  {mpmath.nstr(res.A[n-1], da):28s}  {mpmath.nstr(res.B[n-1], db)}")
        return

    if args.cmd == "eval":
        coeffs = _parse_complex_list(args.coeffs)
        args.num_terms = len(coeffs)
        res = _build_from_args(args)
        val, verr = res.evaluate_with_error([mpc(c) for c in coeffs])
        d = _sig_digits(val, verr, res.working_precision)
        print(str(res))
        print(f"\nL({mpmath.nstr(res.s, 12).strip('()')}) = {mpmath.nstr(val, d)}")
        return


if __name__ == "__main__":
    main()
