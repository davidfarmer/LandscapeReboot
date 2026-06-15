# A smoothed approximate functional equation (Rubinstein, Theorem 1)

`afe.py` computes a **smoothed approximate functional equation** for a completed
L-function `Λ(s)`, following

> M. Rubinstein, *Computational methods and experiments in analytic number
> theory*, [arXiv:math/0412181](https://arxiv.org/abs/math/0412181), **Theorem 1**
> (equations (25)–(26)),

with two changes requested for this task:

1. **A different weight function.** Rubinstein uses `g(s) = δ^{-s}`. Here the
   weight is

   ```
   g(s) = s^m · exp(i·β·s + α·s²)          (m a non-negative integer)
   ```

   Because this `g` is not a pure exponential, the auxiliary functions `f₁, f₂`
   of Theorem 1 do **not** reduce to incomplete Γ-functions. They are evaluated
   directly as Mellin–Barnes contour integrals, numerically, to a requested
   precision.

2. **LMFDB `Γ_ℝ`/`Γ_ℂ` data.** The functional-equation data is supplied in the
   normalization used by the
   [LMFDB](https://www.lmfdb.org/knowledge/show/lfunction.functional_equation).

The output is a **symbolic expression for `L(s)`** that is linear in the unknown
Dirichlet coefficients, with the real and imaginary parts `xₙ = Re b(n)` and
`yₙ = Im b(n)` left as free symbols and everything else reduced to high-precision
numbers. Internally the smoothed AFE produces `Λ(s)·g(s)`; the code divides by the
gamma factor and the weight to return `L(s)` (so the result is independent of the
choice of `g` — see §2).

---

## 1. Conventions

### The L-function (LMFDB normalization)

```
L(s) = Σ_{n≥1} b(n) / n^s ,
```

with completed L-function

```
Λ(s) = N^{s/2} · ∏_{j=1}^{d₁} Γ_ℝ(s + μ_j) · ∏_{k=1}^{d₂} Γ_ℂ(s + ν_k) · L(s),
```

where

```
Γ_ℝ(s) = π^{-s/2} Γ(s/2),        Γ_ℂ(s) = 2 (2π)^{-s} Γ(s) = Γ_ℝ(s) Γ_ℝ(s+1),
```

and functional equation

```
Λ(s) = ε · conj( Λ(1 - conj(s)) ).
```

Here `N` is the conductor, `ε` the root number (`|ε| = 1`), `μ_j` and `ν_k` the
spectral parameters, and the degree is `d = d₁ + 2 d₂`. `Λ(s)` is allowed simple
poles at points `s_k` with residues `r_k`.

### The weight function

```
g(s) = s^m · exp(i·β·s + α·s²),   m ∈ ℤ_{≥0},   α, β ∈ ℂ.
```

(The summation index is called `n`; the exponent in `g` is written `m` to avoid a
clash with Rubinstein's `s^n` — they are the same object.)

---

## 2. The formula that is implemented

Rubinstein states Theorem 1 in the normalization

```
Λ_R(s) = Q^s · ∏_{i} Γ(κ_i s + λ_i) · L(s),      Λ_R(s) = ω · conj(Λ_R(1 - conj s)).
```

We convert the LMFDB data to this form (derivation below):

```
each Γ_ℝ(s+μ_j) → κ = 1/2, λ = μ_j/2
each Γ_ℂ(s+ν_k) → κ = 1,   λ = ν_k

Q = √N · π^{-d₁/2} · (2π)^{-d₂}                 (real, positive)
C = 2^{d₂} · π^{-(Σ_j μ_j)/2} · (2π)^{-(Σ_k ν_k)}
ω = ε · conj(C) / C                              (so Λ = C·Λ_R)
```

Theorem 1, multiplied back through by `C`, gives the smoothed AFE for `Λ(s)·g(s)`
(the code computes this internally, then divides out the gamma factor and weight
to return `L(s)` — see "Solving for `L(s)`" below):

```
Λ(s) g(s) =  Σ_k  r_k g(s_k) / (s - s_k)                          (pole term)

          +  C · Q^s     · Σ_{n≥1}  b(n)      / n^s     · f₁(s, n)

          +  C · ω · Q^{1-s} · Σ_{n≥1} conj(b(n)) / n^{1-s} · f₂(1-s, n)
```

with the Mellin–Barnes integrals

```
f₁(s, n)   = 1/(2πi) ∫_{(ν₀)} [ ∏_i Γ(κ_i (z+s)   + λ_i)      ] · z^{-1} · g(s+z) · (Q/n)^z dz
f₂(1-s, n) = 1/(2πi) ∫_{(ν₀)} [ ∏_i Γ(κ_i (z+1-s) + conj(λ_i)) ] · z^{-1} · g(s-z) · (Q/n)^z dz
```

The contour `Re(z) = ν₀` is placed to the right of `z = 0` and of every pole of
the Γ-factors; the value of each integral is independent of the exact `ν₀` by
Cauchy's theorem.

### Evaluation: a fixed-grid Riemann sum (Poisson summation)

Each integral is evaluated by the **trapezoidal / Riemann sum** on the line
`z = ν₀ + i k h`. For an analytic, rapidly-decaying integrand this is spectrally
accurate — the Riemann-sum error is the sum of the integrand's Fourier transform
over the dual lattice, which is exponentially small in `2π d / h` (`d` = distance
from the contour to the nearest singularity); this is exactly Rubinstein's §2.4
Poisson-summation observation. The step `h` and width `W` are sized once from `d`
and the decay rate to hit the working precision — a single pass, no adaptive
refinement.

Crucially, **only the factor `(Q/n)^z = Q^z·n^{-z}` depends on `n`.** Writing the
`n`-independent part `H(z) = [∏ Γ(…)]·z^{-1}·g(…)·Q^z` and sampling it **once** on
the grid,

```
f(s, n) = (h / 2π) · n^{-ν₀} · Σ_k H(ν₀+i k h) · e^{-i k h ln n}.
```

So the expensive Γ/g evaluations are shared across all Dirichlet terms: the whole
sum costs `O(nodes)` Γ-evaluations instead of `O(terms × nodes)`. This is the
source of the large speed-up over a per-term adaptive quadrature (see
"Performance" below). The previous adaptive-trapezoid implementation is kept in
`afe_trapezoid_backup.py`.

### Solving for `L(s)`

Since `Λ(s) = 𝒢(s)·L(s)` with the complete gamma factor

```
𝒢(s) = N^{s/2} ∏ⱼ Γ_ℝ(s+μⱼ) ∏ₖ Γ_ℂ(s+νₖ) = C · Q^s · ∏ᵢ Γ(κᵢ s + λᵢ),
```

dividing the boxed identity by the scalar `D = g(s)·𝒢(s)` gives `L(s)` directly:

```
L(s) = Λ(s) g(s) / ( g(s) · 𝒢(s) ).
```

The explicit weight `g(s)` **cancels**, so the returned `L(s)` does not depend on
the choice of `g` (a useful internal check; `g` still controls convergence).

### Symbolic output

Writing `b(n) = xₙ + i·yₙ` and the (already `D`-divided) per-term coefficients

```
Aₙ = [ C · Q^s · n^{-s}      · f₁(s, n)   ] / D     (coefficient of b(n)),
Bₙ = [ C · ω · Q^{1-s} · n^{s-1} · f₂(1-s, n) ] / D (coefficient of conj b(n)),
```

the returned expression is

```
L(s) = (pole term)/D  +  Σ_{n=1}^{M} [ (Aₙ + Bₙ)·xₙ  +  i(Aₙ - Bₙ)·yₙ ],
```

a sympy expression linear in the real symbols `xₙ, yₙ`, with `Aₙ`, `Bₙ`, and the
pole term carried as high-precision complex numbers. `M` (`num_terms`) is chosen
automatically from the requested accuracy.

When printed, each number is shown to **its own actual precision** — the number
of significant figures estimated to be correct from the computation's error
(the achieved contour-integral accuracy, round-off at the working precision, and,
for the summed `L(s)` value, the truncation tail), capped at the working
precision. This is *not* tied to the `accuracy` request: a coefficient that
converges to full working precision is shown to that many digits, while a tiny
near-noise coefficient is shown to only the digits that mean anything. Pass an
explicit `digits=` / `--digits` to force a fixed count.

---

## 3. Admissibility of `g`

Theorem 1 requires `g` entire with `|Λ(z+s) g(z+s) z^{-1}| → 0` as `|Im z| → ∞`
in vertical strips. For `g(s) = s^m exp(iβs + αs²)`, using that the Γ-factors
decay like `exp(-(π/2)(Σ κ_i)|Im z|)`:

- **`Re(α) > 0`** — `g` itself has Gaussian decay `exp(-Re(α)·(Im z)²)`. Always
  admissible (this is the recommended, well-conditioned regime, and gives the
  fastest convergence).
- **`Re(α) = 0`** — admissible iff `(π/2)·Σ κ_i > |Re(β) + 2·Im(α)·u|` for all
  `u = Re(s+z)` in the strip, i.e. the Γ-decay must beat the linear-exponential
  growth of `g`. (`g ≡ 1`, i.e. `m=α=β=0`, is the classical case and is always
  admissible, but converges only as fast as the Γ-factors allow.)
- **`Re(α) < 0`** — `g` grows like a Gaussian; never admissible.

`afe()` checks this and records the verdict in `result.info["g_admissible"]`; the
CLI prints a warning if it fails.

---

## 4. Precision and accuracy

Two independent knobs:

| knob | meaning |
|------|---------|
| `accuracy` | target for **truncation** of the Dirichlet sum: terms below `10^{-accuracy}` are dropped, which fixes the number of terms `M`. It does *not* set the display precision. |
| `working_precision` | the internal `mpmath` precision (`mp.dps`), and the target the contour integrals are pushed to. Defaults to `accuracy + 15 + 3·degree`. |

The number of significant figures *printed* for each value is determined from its
actual computed error (see "Symbolic output"), not from `accuracy`.

The number of Dirichlet terms `M` is grown until `(|Aₙ| + |Bₙ|)·n^r < 10^{-accuracy}`
for several consecutive `n`, where `r = coeff_growth` is the assumed bound
`|b(n)| = O(n^r)` (default `r = 0`, appropriate for the analytic normalization;
raise it for unnormalized coefficients). You may also force `num_terms` directly.

---

## 5. Usage

### Python API

```python
from afe import afe

# Riemann zeta at s = 2 + 3i, weight g(s) = s^2 exp(0.5 i s + 0.2 s^2)
res = afe(
    s=2+3j,
    mu=[0], nu=[], N=1, epsilon=1,        # ζ: one Γ_ℝ(s), conductor 1, sign +1
    poles=[(1, 1), (0, -1)],              # residues of Λ_ζ at s=1 and s=0
    g_m=2, g_alpha=0.2, g_beta=0.5,
    accuracy=30,
)

print(res)                                # summary
expr = res.symbolic_expression()          # sympy expression for L(s) in x1,y1,x2,y2,...
print(expr)

# Plug in numeric coefficients (here b(n)=1, i.e. ζ) and recover L(s)=ζ(s):
val = res.evaluate([1]*res.num_terms)
```

`res.symbolic_expression()` returns a genuine `sympy` object, so you can
`subs`, differentiate, collect, `lambdify`, etc.

### Substituting known coefficients

`afe_substitute(s, coefficients, **same_kwargs_as_afe)` returns `L(s)` with a
*partial* set of coefficients plugged in and the rest left symbolic. The known
coefficients are a dict `{n: b(n)}` or a sequence `[b(1), b(2), …]` (use `None`
to leave an entry symbolic); indices past the truncation length are ignored.

```python
from afe import afe_substitute

# Riemann zeta at 2+3i, with b(1)=b(2)=1 fixed and b(3), b(4), … left symbolic:
expr, res = afe_substitute(2+3j, {1: 1, 2: 1}, mu=[0], N=1, epsilon=1,
                           poles=[(1, 1), (0, -1)], accuracy=12)
print(expr)        # numeric constant + x3,y3,x4,y4,… terms

# Supplying every coefficient collapses to a single number, L(s):
full = {n: 1 for n in range(1, res.num_terms + 1)}
expr_full, _ = afe_substitute(2+3j, full, mu=[0], N=1, epsilon=1,
                              poles=[(1, 1), (0, -1)], accuracy=12)
print(complex(expr_full))     # ≈ zeta(2+3i)
```

Equivalently, `res.symbolic_expression(known=...)` or `res.substitute(known)` on
an existing result. From the CLI, add `--known` to `expr`:

```bash
# b(1)=b(2)=b(3)=1 substituted, the rest symbolic:
python3 afe.py expr --s "2+3j" --mu "0" --N 1 --epsilon 1 --poles "1:1;0:-1" \
        --accuracy 10 --known "1,1,1"
# sparse form (only b(1) and b(5)):  --known "1:1,5:2"
```

### A pole-free relation among the coefficients

`coefficient_relation(s, …)` takes the *same arguments* as `afe` but uses the
**same contour integral with the `z^{-1}` factor removed**, so the integrand has
no pole at `z=0`. The closed contour integral of `Λ(z+s)g(z+s)` is then just the
sum of residues at the poles of `Λ` (zero for an entire L-function), giving a
linear relation the coefficients must satisfy:

```
Σ_n [ Ã_n·b(n) + B̃_n·conj(b(n)) ]  =  Σ_k r_k g(s_k),
```

where `Ã_n, B̃_n` are the AFE coefficients built from the pole-free integrals
(`B̃_n` carries an extra minus sign — removing `z^{-1}` drops the sign flip it
produced under `z→−z` in the functional-equation step). The returned `AFEResult`
represents *left − right*: its `symbolic_expression()` is an expression in the
`x_n, y_n` that is **identically zero** for the true coefficients, and its
`evaluate(coeffs)` returns a value `≈0`. Because it needs no reference value, it
is a self-contained consistency test (and could anchor a least-squares solve for
unknown coefficients).

```python
from afe import coefficient_relation, elliptic_curve_11a_bcoeffs

r = coefficient_relation(0.6+1.2j, mu=[], nu=[0.5], N=11, epsilon=1, accuracy=15)
print(abs(r.evaluate(elliptic_curve_11a_bcoeffs(r.num_terms))))   # ~1e-20  (≈ 0)
```

`g` must satisfy `|Λ(z+s)·g(z+s)| → 0` in vertical strips (slightly stronger than
the AFE's `|Λg/z| → 0`). The relation depends on the chosen `s` and `g`; each gives
a different valid constraint.

### Command line

```bash
# Validate the implementation against ζ and Dirichlet L-functions:
python3 afe.py selftest --accuracy 25

# Print the symbolic expression for a degree-2 example (a weight-k cusp form
# has one Γ_ℂ factor); here ν = (k-1)/2, say k = 12 -> ν = 5.5, N = 1, sign +1.
# (g = 1 here keeps it to a handful of terms.)
python3 afe.py expr --s "0.5+10j" --nu "5.5" --N 1 --epsilon 1 \
        --m 0 --alpha 0 --beta 0 --accuracy 15 --table

# Riemann zeta, symbolic, with the requested weight:
python3 afe.py expr --s "2+3j" --mu "0" --N 1 --epsilon 1 \
        --poles "1:1;0:-1" --m 2 --alpha 0.2 --beta 0.5 --accuracy 25

# Evaluate at explicit coefficients (defines the number of terms):
python3 afe.py eval --s "2+3j" --mu "0" --N 1 --epsilon 1 \
        --poles "1:1;0:-1" --m 2 --alpha 0.2 --beta 0.5 --coeffs "1,1,1,1,1,1,1,1"
```

### Worked example: an elliptic curve L-function (LMFDB)

An elliptic curve `E/ℚ` gives a degree-2, self-dual L-function. In LMFDB
analytic normalization (`b(n) = a_n/√n`, functional equation `s ↔ 1-s`) the data
is a single `Γ_ℂ(s + ½)` factor — so `mu = []`, `nu = [1/2]` — with conductor
`N` and root number `ε`.

Taking the curve **11.a** ([LMFDB L-function
`2-11-1.1-c1-0-0`](https://www.lmfdb.org/L/2/11/1.1/c1/0/0)): `N = 11`,
`ν = 1/2`, `ε = +1`, analytic rank 0, and the Frobenius traces `a_p` read off the
LMFDB Euler factors `1 - a_p X + p X²` (`a₂=-2, a₃=-1, a₅=1, a₇=-2, a₁₁=1,
a₁₃=4, a₁₇=-2, a₁₉=0, …`). These are bundled in `afe.py` as `EC_11A` /
`elliptic_curve_11a_bcoeffs`.

```python
import mpmath
from mpmath import mp, mpf
from afe import afe, elliptic_curve_11a_bcoeffs

mp.dps = 30
b = elliptic_curve_11a_bcoeffs(120)          # analytic coeffs b(n) = a_n/sqrt(n)

# AFE at the central point s = 1/2 (g = 1); evaluate() returns L(s) directly:
r = afe(mpf(1)/2, mu=[], nu=[mpf(1)/2], N=11, epsilon=1, accuracy=13)
L_half = r.evaluate(b[:r.num_terms])         # = L(1/2)
print(L_half.real)        # 0.2538418608559107  == LMFDB central value, 21 terms
```

This reproduces the LMFDB central value `L(1/2) = 0.2538418608559107` to `~10⁻¹⁶`.
For the symbolic expression at another point, via the CLI (coefficients left as
`xₙ, yₙ`):

```bash
python3 afe.py expr --s "0.5+2j" --nu "0.5" --N 11 --epsilon 1 \
        --m 0 --alpha 0 --beta 0 --accuracy 12
```

CLI argument reference (subcommands `expr`, `eval`, `selftest`):

| flag | meaning |
|------|---------|
| `--s` | evaluation point, e.g. `0.5+14.1j` |
| `--mu`, `--nu` | comma-separated `Γ_ℝ` / `Γ_ℂ` shifts, e.g. `--mu "0,1"` |
| `--N`, `--epsilon` | conductor and root number |
| `--poles` | poles & residues of `Λ`, formatted `s1:r1;s2:r2` |
| `--m`, `--alpha`, `--beta` | parameters of `g(s)=s^m exp(iβs+αs²)` |
| `--accuracy`, `--working-precision` | output digits / internal `mp.dps` |
| `--num-terms`, `--coeff-growth`, `--contour-nu` | truncation / contour overrides |
| `--digits`, `--table` (`expr`) | force fixed significant digits per coefficient (default: each value's actual precision); also dump `Aₙ,Bₙ` |
| `--coeffs` (`eval`) | comma list `b(1),b(2),…` |

---

## 6. Worked conversion identity (derivation)

Starting from the LMFDB form and substituting `Γ_ℝ(s+μ)=π^{-(s+μ)/2}Γ((s+μ)/2)`
and `Γ_ℂ(s+ν)=2(2π)^{-(s+ν)}Γ(s+ν)`:

```
Λ(s) = N^{s/2} ∏_j π^{-(s+μ_j)/2} Γ((s+μ_j)/2)
              ∏_k 2 (2π)^{-(s+ν_k)} Γ(s+ν_k) · L(s).
```

Collect the `s`-dependent exponential base into `Q^s`:

```
Q^s = N^{s/2} · π^{-d₁ s/2} · (2π)^{-d₂ s}   ⇒   Q = √N · π^{-d₁/2} · (2π)^{-d₂}.
```

The remaining `s`-independent constant is

```
C = ∏_j π^{-μ_j/2} · ∏_k 2(2π)^{-ν_k} = 2^{d₂} · π^{-(Σμ_j)/2} · (2π)^{-(Σν_k)}.
```

The Γ-arguments become `Γ(½ s + μ_j/2)` (so `κ=½, λ=μ_j/2`) and `Γ(s + ν_k)`
(so `κ=1, λ=ν_k`). Hence `Λ(s) = C · Λ_R(s)`. Feeding
`Λ(s) = ε·conj(Λ(1-conj s))` through this gives the Rubinstein root number

```
ω = ε · conj(C) / C        (|ω| = 1; ω = ε when C is real).
```

Applying Theorem 1 to `Λ_R` and multiplying by `C` yields the boxed formula in
§2; the residue of `Λ_R` at `s_k` times `C` is exactly the residue `r_k` of the
LMFDB-normalized `Λ`, so the pole term is simply `Σ_k r_k g(s_k)/(s-s_k)`.

---

## 7. Validation

`python3 afe.py selftest --accuracy D` runs ten checks:

1. **`f₁` vs the closed form.** With `g = 1`, `f₁(s,n)` must equal the upper
   incomplete Gamma `Γ(s/2, πn²)` for Riemann ζ — checked at `s = 2+3i`
   (absolute error `~10⁻¹⁷`).
2. **ζ AFE, `g = 1`, `s = 2+3i`** (degree 1, poles at `s = 0,1`), against
   `mpmath.zeta` — rel. error `~10⁻¹⁶`.
3. **A Dirichlet L-function** for a *complex* primitive (odd) character `χ mod 5`
   of order 4, at `s = 0.7 + 1.1i`. This exercises a nontrivial conductor
   `N = 5`, an odd `Γ_ℝ` shift `μ = 1`, a complex root number `ε = τ(χ)/(i√5)`,
   and — because the coefficients are genuinely complex — the `xₙ`/`yₙ` split via
   `conj(b(n))`. The independent value uses `L(s,χ) = q^{-s} Σ_a χ(a) ζ(s, a/q)`.
4. **The requested Gaussian weight** `g(s) = s² exp(0.5·i·s + 0.2·s²)` for ζ at
   `s = 2+3i` (slower convergence; checked to 12 digits).
5. **Degree-2 / `Γ_ℂ` consistency.** Because `Γ_ℂ(s+ν) = Γ_ℝ(s+ν)·Γ_ℝ(s+ν+1)`
   (Legendre duplication), the per-term coefficients must be identical whether
   the data is supplied as one `Γ_ℂ(ν)` or as two `Γ_ℝ(ν), Γ_ℝ(ν+1)`; agreement
   is to full working precision (`~10⁻³⁴`).
6. **Elliptic curve 11.a against the LMFDB.** Using the LMFDB data for L-function
   `2-11-1.1-c1-0-0` (`N=11`, `ν=1/2`, `ε=+1`, coefficients from the LMFDB Euler
   factors), the AFE at `s=1/2` reproduces the LMFDB central value
   `L(1/2) = 0.2538418608559107` to `~10⁻¹⁶` (21 terms).
7. **Dirichlet L-function, complex *even* character.** A primitive order-3
   character `χ mod 7` (so `χ(−1)=1`, hence `μ=0`, and complex root number
   `ε = τ(χ)/√7`), at `s = 1.3 + 0.4i`, against the Hurwitz-zeta value. This
   complements check 3 (an *odd* complex character, `μ=1`) by exercising the even
   case; both are non-self-dual, so the dual `conj(b(n)) = χ̄(n)` is used.
8. **Maass form L-function.** `L(s) = ζ(s+iR)·ζ(s−iR)` with `R = 9.53369526135`
   (the spectral parameter of the first SL(2,ℤ) Maass cusp form, from the LMFDB).
   This is the L-function of the Eisenstein series `E(z, ½+iR)` — a degree-2,
   self-dual Maass form with conductor 1, `ε=+1`, and **imaginary** spectral
   parameters `μ = ±iR`, i.e. archimedean factor `Γ_ℝ(s+iR)Γ_ℝ(s−iR)`, exactly the
   gamma factors of a cuspidal Maass form. Its completed `Λ(s)=ξ(s+iR)ξ(s−iR)` has
   four poles (`s = ±iR, 1±iR`). The AFE reproduces the exact value
   `ζ(s+iR)ζ(s−iR)` to `~10⁻¹⁸` (checked at `s=2+i`). This exercises imaginary
   spectral shifts — the integration grid must be re-centred on the integrand's
   mass near `τ ≈ ±R` rather than `τ ≈ 0`.

   *(Note: a cuspidal Maass form would be entire with transcendental
   coefficients; the Eisenstein form is used here because its coefficients
   `b(n)=Σ_{d∣n}(n/d²)^{iR}` and value are known in closed form, making the test
   exact and self-contained while exercising the identical archimedean type.)*
9. **Primitive non-arithmetic degree-3 L-function** — the GL(3) Maass cusp form
   [`3-1-1.1-r0e3-m0.17m16.40p16.57-0`](https://www.lmfdb.org/L/3/1/1.1/r0e3/m0.17m16.40p16.57/0)
   from the LMFDB: degree 3, conductor 1, **primitive**, **non-self-dual**, with
   three imaginary spectral shifts `μ = i·(−0.171, −16.403, 16.574)` (so the
   integrand mass sits near `τ ≈ ±16.5`). Coefficients are built from the LMFDB
   Euler factors `F_p(X)=1+c₁X−\overline{c₁}X²−X³` (transcendental, complex), and
   the test checks that `L` **vanishes at the LMFDB zeros** `½+iγ_k` (first two),
   reaching `~10⁻¹³` (the precision floor of the 13-digit `γ_k`). This is the only
   degree-3 / non-self-dual / primitive case, and the only one validated against
   zeros rather than a value.
10. **`coefficient_relation` is zero.** The pole-free relation (see §9 of usage)
    must vanish identically when the true coefficients are supplied. Checked on an
    entire L-function with real coefficients (elliptic curve 11.a, `~10⁻²⁰`), one
    with complex coefficients (the degree-3 GL(3) Maass form, `~10⁻¹⁶`), and ζ with
    `g(s)=s` so that the residue term is nonzero (`~10⁻¹⁷`).

### A note on convergence and the choice of `g`

The number of Dirichlet terms is governed by how fast `f₁, f₂` decay in `n`:

- **`Re(α) = 0`** (Rubinstein's regime, e.g. `g ≡ 1` or the `δ^{-s}` phase
  trick): `fₙ` decays at the incomplete-Gamma rate `exp(-c·n^{2/d})` — only a few
  terms are needed.
- **`Re(α) > 0`** (a genuine Gaussian factor): the Gaussian gives excellent decay
  in the *vertical* direction (so `Λ(s)g(s)` is well-behaved for large `|Im s|`),
  but it slows the decay in `n` to roughly `exp(-c·n)`, so many more terms are
  needed for the same accuracy. This is a mathematical property of the weight,
  not a limitation of the code; budget accordingly (or supply `--num-terms`).

## Performance

Switching from per-term adaptive quadrature to the shared fixed-grid Riemann sum
(above) reuses the costly Γ/g evaluations across all Dirichlet terms. Measured
wall-clock (same machine, identical inputs and results):

| case | adaptive trapezoid | fixed-grid Riemann sum | speed-up |
|------|-------------------:|-----------------------:|---------:|
| ζ, `g=1`, `s=2+3i`, 20-digit (7 terms, wp 38) | 68.6 s | 4.5 s | **15×** |
| ζ, Gaussian `g`, 80 terms, wp 25 | 118.7 s | 1.7 s | **71×** |
| Elliptic curve 11.a (degree 2), `s=½`, 15-digit (23 terms, wp 36) | 110.0 s | 3.2 s | **34×** |
| full `selftest --accuracy 12` | 142 s | 5.5 s | **26×** |

The speed-up grows with the number of terms (the Gaussian case has the most), since
the per-term cost drops from a full quadrature to one weighted sum. Results are
unchanged to working precision. The old method is preserved in
`afe_trapezoid_backup.py`.

---

## 8. Requirements

- Python 3
- [`mpmath`](https://mpmath.org/) (arbitrary-precision arithmetic and Γ-function)
- [`sympy`](https://www.sympy.org/) (only for `symbolic_expression()` and the
  `expr` CLI subcommand)

The algorithm is written in plain `mpmath` and ports directly to Sage
(`Sage`'s `ComplexField`/`pari` could replace the numerics if desired).
