#!/usr/bin/env python3
r"""
families.py -- the L-function "landscapes" the search engine (lsearch.py) operates
on, and the plain-text registry that describes them.

A *landscape* (a tempered balanced analytic L-function family, per arXiv:1711.10375
section 2.1) is fully specified by:

  * its Gamma factors -- signature (d1, d2) and, per factor, a real Selberg shift
    (delta in {0,1} for Gamma_R, kappa in {1/2,1,...} for Gamma_C) plus the
    imaginary part written as a combination of the free spectral parameters
    lambda_1..lambda_r (the search dimension);
  * conductor N and central character chi mod N;
  * the good-prime Euler-factor model: the Satake roots lie on |z|=1 with product
    chi(p), so the local factor is (twisted) self-reciprocal and is determined by
    its first floor((d-1)/2) Dirichlet coefficients -- 1 unknown per good prime for
    degree 3 and 4, two for degree 5 and 6.  The self_dual flag picks the twisted
    (conjugate; generic GL(3)) vs untwisted (self-dual) relation;
  * (later) bad-prime local factors and the local-sign relations.

These come from the registry file ``landscapes.txt`` (or an ad-hoc file in the same
format); this module parses one into the `Landscape` + `EulerProduct` pair that the
engine consumes, bundled as a `Family`.  The engine itself is family-agnostic: it
only calls ``landscape.mu_from_point`` / ``euler.bcoeffs_from_ap`` and friends.

The per-prime unknowns are LISTS of complex numbers (the first floor((d-1)/2)
Dirichlet coefficients b(p), b(p^2), ...); for the GL(3) family the list has length
one and equals [a_p], so this reduces exactly to the original degree-3 code.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import mpmath
from mpmath import mpf, mpc


# ---------------------------------------------------------------------------
# Prime helpers (shared by the Euler builder here and the sizing in lsearch.py)
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
# Central characters
# ---------------------------------------------------------------------------
#
# For now only the trivial character is implemented (chi(p) = 1 for all p, the
# conductor-1 case).  A Conrey label q.n will later build the Dirichlet character
# mod q; the only thing the Euler builder needs is chi(p) as a complex number.

class Character:
    """A Dirichlet character chi mod N, callable as chi(n) -> mpc."""
    def __init__(self, modulus, label, fn):
        self.modulus = modulus
        self.label = label
        self._fn = fn

    def __call__(self, n):
        return self._fn(n)


def parse_character(spec, modulus):
    spec = (spec or "trivial").strip()
    if spec in ("trivial", "", "1", "%d.1" % modulus):
        return Character(modulus, "trivial", lambda n: mpc(1))
    raise NotImplementedError(
        "only the trivial character is implemented so far (got %r); a Conrey "
        "label q.n will be added with the conductor>1 work" % spec)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Landscape:
    """A functional-equation family: known conductor, Gamma factors with unknown
    spectral parameters.  A 'point' is the tuple of free spectral parameters."""
    name: str
    degree: int
    conductor: int
    dim: int                         # number of free spectral parameters
    mu_from_point: Callable          # point -> list of Gamma_R shifts (mu), complex
    nu_from_point: Callable          # point -> list of Gamma_C shifts (nu), complex
    signature: tuple = (0, 0)        # (d1, d2)


@dataclass
class EulerProduct:
    """The Euler-product shape: how the full Dirichlet coefficient vector is built
    from the independent per-prime unknowns, and how to read those unknowns back.

    The per-prime unknown is a LIST of `complex_unknowns_per_prime` complex numbers
    (the first few Dirichlet coefficients b(p), b(p^2), ...)."""
    name: str
    degree: int
    complex_unknowns_per_prime: int
    bcoeffs_from_ap: Callable        # (ap: {p: [b(p),...]}, M) -> [b(1), ..., b(M)]
    extract_ap: Callable             # ([b(1..M)]) -> {p: [b(p), b(p^2), ...]}

    @property
    def real_unknowns_per_prime(self):
        return 2 * self.complex_unknowns_per_prime


@dataclass
class Family:
    """A registered landscape: the engine-facing Landscape + EulerProduct, plus the
    descriptive metadata parsed from the registry."""
    name: str
    description: str
    landscape: Landscape
    euler: EulerProduct
    conductor: int
    signature: tuple
    character: Character
    self_dual: bool
    bad: list = field(default_factory=list)   # bad-prime data (empty for now)


@dataclass
class KnownTarget:
    """A fully known L-function sitting in a landscape, used as ground truth."""
    name: str
    landscape: Landscape
    euler: EulerProduct
    point: tuple                     # the true spectral parameters
    epsilon: mpc                     # the true sign
    ap: dict                         # the true {p: [b(p), b(p^2), ...]}


# ---------------------------------------------------------------------------
# Good-prime Euler-factor model: (twisted) self-reciprocal local factor
# ---------------------------------------------------------------------------
#
# The local factor is  F_p(z) = sum_{j=0}^{d} c_j z^j  with c_0 = 1.  Temperedness
# puts the Satake roots on |z| = 1 with product chi(p) (Ax4b, Ax5a), which forces
# the self-reciprocity
#
#       c_{d-k} = (-1)^d chi(p) conj(c_k)            (k = 0, ..., d).
#
# So the independent unknowns are c_1, ..., c_r with r = floor((d-1)/2); we carry
# them as the first r Dirichlet coefficients b(p), b(p^2), ..., b(p^r) (related to
# the c's by 1/F_p(z) = sum_k b(p^k) z^k), because that is what the rest of the
# search reports and seeds.  For degree 3 this is the single a_p = b(p) and the
# whole thing reduces to  1 - a_p z + conj(a_p) z^2 - z^3.

def _local_coeffs_from_bp(bp_list, degree, chi, self_dual):
    """Reconstruct the local-factor coefficients c_0..c_d from the first r Dirichlet
    coefficients bp_list = [b(p), b(p^2), ..., b(p^r)] and the (twisted) self-reciprocity
    c_{d-k} = (-1)^d chi(p) * (c_k if self_dual else conj(c_k))."""
    d = degree
    r = (d - 1) // 2
    b = [mpc(1)] + [mpc(x) for x in bp_list]          # b[0]=1, b[1..r]
    c = [mpc(0)] * (d + 1)
    c[0] = mpc(1)
    # 1/F = sum b z^k  =>  sum_{i=0}^{k} c_i b(p^{k-i}) = 0  (k>=1)  =>  c_k below
    for k in range(1, r + 1):
        s = b[k]
        for i in range(1, k):
            s += c[i] * b[k - i]
        c[k] = -s
    sign = mpc((-1) ** d)
    for j in range(0, r + 1):                          # fill the reciprocal half
        cj = c[j] if self_dual else mpmath.conj(c[j])
        c[d - j] = sign * chi * cj
    return c


def _bppow_list(bp_list, degree, chi, self_dual, kmax):
    """[b(p^0), ..., b(p^kmax)] from the unknowns bp_list (= [b(p),...,b(p^r)])."""
    d = degree
    c = _local_coeffs_from_bp(bp_list, d, chi, self_dual)
    bb = [mpc(1)]
    for k in range(1, int(kmax) + 1):
        s = mpc(0)
        for i in range(1, min(d, k) + 1):
            s += c[i] * bb[k - i]
        bb.append(-s)
    return bb


def euler_local(degree, character, self_dual=False):
    """EulerProduct for a tempered good-prime factor.  The Satake roots lie on |z|=1
    with product chi(p), so the local factor is (twisted) self-reciprocal and is fixed
    by its first r = floor((d-1)/2) Dirichlet coefficients [b(p), ..., b(p^r)] (the
    per-prime unknown).  `self_dual` chooses the relation: True (self-dual) uses
    c_{d-k} = (-1)^d chi(p) c_k, False (e.g. generic GL(3)) the conjugate version."""
    d = int(degree)
    if d < 1:
        raise ValueError("degree must be >= 1")
    if d % 2 == 0:
        # even degree has a middle coefficient c_{d/2} constrained only to a real
        # 1-parameter line (c_{d/2} = (-1)^d chi(p) conj(c_{d/2})); that extra real
        # unknown is not carried yet -- pin it against a known even-degree example.
        raise NotImplementedError(
            "even-degree local factor (middle coefficient) not implemented yet; pin "
            "against a known degree-%d example before enabling" % d)
    r = (d - 1) // 2

    def bcoeffs_from_ap(ap, M):
        M = int(M)
        spf = _smallest_prime_factor_table(M)
        # cache the prime-power coefficients we need
        b = [mpc(0)] * (M + 1)
        if M >= 1:
            b[1] = mpc(1)
        powcache = {}
        for n in range(2, M + 1):
            m, val = n, mpc(1)
            while m > 1:
                p = spf[m]
                e = 0
                while m % p == 0:
                    m //= p
                    e += 1
                bp = ap.get(p)
                if bp is None:               # below the noise level -> drop these n
                    val = mpc(0)
                    break
                pc = powcache.get(p)
                if pc is None or len(pc) <= e:
                    pc = _bppow_list(bp, d, character(p), self_dual, e)
                    powcache[p] = pc
                val *= pc[e]
            b[n] = val
        return b[1:M + 1]

    def extract_ap(b):
        M = len(b)
        out = {}
        for p in primes_up_to(M):
            vals = []
            pk = p
            for _ in range(r):
                if pk <= M:
                    vals.append(mpc(b[pk - 1]))
                    pk *= p
                else:
                    vals.append(mpc(0))
            out[p] = vals
        return out

    return EulerProduct(
        name="degree %d, tempered, %sself-reciprocal" %
             (d, "" if self_dual else "twisted-"),
        degree=d, complex_unknowns_per_prime=r,
        bcoeffs_from_ap=bcoeffs_from_ap, extract_ap=extract_ap)


# ---------------------------------------------------------------------------
# Gamma-spec parser:  "R(delta | c1, c2, ...)" / "C(kappa | c1, c2, ...)"
# ---------------------------------------------------------------------------

def _num(tok):
    """Parse a real shift/coefficient: decimal ('0.5', '-1') or a fraction ('-1/2')."""
    tok = tok.strip()
    if "/" in tok:
        num, den = tok.split("/")
        return mpf(num.strip()) / mpf(den.strip())
    return mpf(tok)


_GAMMA_RE = re.compile(r"^\s*([RC])\s*\(\s*([^|]+?)\s*\|\s*(.*?)\s*\)\s*$")


def parse_gamma_factor(line):
    """Parse one gamma line into (kind 'R'/'C', shift, [lambda coefficients])."""
    m = _GAMMA_RE.match(line)
    if not m:
        raise ValueError("bad gamma factor %r (expected 'R(delta | c1, c2, ...)')" % line)
    kind = m.group(1)
    shift = _num(m.group(2))
    coeffs = [_num(t) for t in m.group(3).split(",") if t.strip() != ""]
    return kind, shift, coeffs


def build_gamma_maps(factors):
    """From parsed gamma factors build (dim, mu_from_point, nu_from_point).

    mu/nu factor k contributes  shift + i * (coeffs . lambda)  to the Gamma_R /
    Gamma_C shift list."""
    dim = max((len(c) for (_k, _s, c) in factors), default=0)

    def shift_value(shift, coeffs, pt):
        val = mpc(shift)
        for k, c in enumerate(coeffs):
            if c != 0:
                val += 1j * mpf(c) * mpf(pt[k])
        return val

    R = [(s, c) for (k, s, c) in factors if k == "R"]
    C = [(s, c) for (k, s, c) in factors if k == "C"]

    def mu_from_point(pt):
        return [shift_value(s, c, pt) for (s, c) in R]

    def nu_from_point(pt):
        return [shift_value(s, c, pt) for (s, c) in C]

    return dim, mu_from_point, nu_from_point


# ---------------------------------------------------------------------------
# Registry parsing
# ---------------------------------------------------------------------------

def _registry_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "landscapes.txt")


def _parse_blocks(text):
    """Split registry text into {name: {key: value or [values for repeated keys]}}."""
    blocks, cur, name = {}, None, None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        head = re.match(r"^\s*\[\s*landscape\s+([^\]]+?)\s*\]\s*$", line)
        if head:
            name = head.group(1).strip()
            cur = {}
            blocks[name] = cur
            continue
        if cur is None:
            raise ValueError("registry line outside any [landscape ...] block: %r" % raw)
        if "=" not in line:
            raise ValueError("registry line without '=': %r" % raw)
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if key == "gamma":
            cur.setdefault("gamma", []).append(val)
        else:
            cur[key] = val
    return blocks


def family_from_block(name, blk):
    """Build a Family from one parsed registry block."""
    degree = int(blk["degree"])
    d1, d2 = (int(x) for x in blk.get("signature", "%d, 0" % degree).split(","))
    if d1 + 2 * d2 != degree:
        raise ValueError("landscape %s: signature (%d,%d) does not match degree %d"
                         % (name, d1, d2, degree))
    conductor = int(blk.get("conductor", "1"))
    character = parse_character(blk.get("character", "trivial"), conductor)
    self_dual = blk.get("self_dual", "false").strip().lower() in ("true", "1", "yes")

    factors = [parse_gamma_factor(g) for g in blk.get("gamma", [])]
    nR = sum(1 for (k, _s, _c) in factors if k == "R")
    nC = sum(1 for (k, _s, _c) in factors if k == "C")
    if (nR, nC) != (d1, d2):
        raise ValueError("landscape %s: gamma factors (%d R, %d C) do not match "
                         "signature (%d, %d)" % (name, nR, nC, d1, d2))
    dim, mu_from_point, nu_from_point = build_gamma_maps(factors)

    landscape = Landscape(name=blk.get("description", name), degree=degree,
                          conductor=conductor, dim=dim, mu_from_point=mu_from_point,
                          nu_from_point=nu_from_point, signature=(d1, d2))

    euler = euler_local(degree, character, self_dual)

    return Family(name=name, description=blk.get("description", name),
                  landscape=landscape, euler=euler, conductor=conductor,
                  signature=(d1, d2), character=character, self_dual=self_dual)


def load_landscape_file(path):
    """Parse a registry / ad-hoc landscape file into {name: Family}."""
    with open(path) as fh:
        blocks = _parse_blocks(fh.read())
    return {nm: family_from_block(nm, blk) for nm, blk in blocks.items()}


_REGISTRY_CACHE = None


def registry():
    """The default registry (landscapes.txt next to this module), parsed once."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = load_landscape_file(_registry_path())
    return _REGISTRY_CACHE


def get_family(name, path=None):
    """Look up a Family by name in the default registry, or in `path` if given."""
    table = load_landscape_file(path) if path else registry()
    if name not in table:
        raise KeyError("no landscape %r (available: %s)"
                       % (name, ", ".join(sorted(table))))
    return table[name]


# ---------------------------------------------------------------------------
# Known ground-truth targets (for the self-tests)
# ---------------------------------------------------------------------------

def gl3_known_target():
    """Ground truth for the R0R0R0N1 landscape: the first SL(3,Z) Maass form, as a
    point with its true sign and per-prime unknowns, built from afe's stored data."""
    from afe import GL3_MAASS, gl3_maass_bcoeffs
    fam = get_family("R0R0R0N1")
    lam = [mpmath.re(-1j * m) for m in GL3_MAASS["mu"]]
    point = (lam[0], lam[1])
    b = gl3_maass_bcoeffs(40)            # n<=40 uses primes up to 37 (all stored)
    ap = fam.euler.extract_ap(b)
    return KnownTarget(name="SL(3,Z) Maass form 3-1-1.1-r0e3-...",
                       landscape=fam.landscape, euler=fam.euler,
                       point=point, epsilon=mpc(1), ap=ap)


# Convenience handles used by lsearch.py's self-tests and CLI default.
GL3_FAMILY = get_family("R0R0R0N1")
GL3_LANDSCAPE = GL3_FAMILY.landscape
GL3_EULER = GL3_FAMILY.euler
