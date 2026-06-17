# Orientation for a new reader

This is a research codebase for **finding L-functions from only their functional
equation and Euler product** (cf. arXiv:1212.4545 and arXiv:2303.00079). This file is
the "things you wish someone had told you" companion to the formal docs. It assumes you
will also read `README.md` (which documents the analytic engine `afe.py` thoroughly) and
the docstrings.

If you read nothing else, read **§3 (mental model)**, **§6 (hard-won knowledge)**, and
**§7 (gotchas)** — those are the parts not written down anywhere else.

---

## 1. The two files

- **`afe.py`** — the analytic engine. Computes a *smoothed approximate functional
  equation* (Rubinstein, Theorem 1 of arXiv:math/0412181) for an L-function given its
  Γ-factors, conductor, sign, and Dirichlet coefficients. Fully documented in
  `README.md`. The piece the search actually uses is **`coefficient_relation` /
  `coefficient_relation_grid`** (see §4).
- **`lsearch.py`** — the search. Given a *landscape* (a functional equation with known
  conductor but unknown spectral parameters) it looks for the spectral parameters
  `(λ1, λ2, …)` and the Euler coefficients `a_p` of an actual L-function living in that
  landscape. This is the newer work and is **not** covered by `README.md` — only by
  docstrings and this file.

Backups (`*_backup.py`, `README_backup.md`) are old snapshots; ignore them.

Everything runs in **mpmath** arbitrary precision. `mp.dps` (decimal digits) is set
explicitly by the search; don't assume the default.

---

## 2. What problem the search solves

A *landscape* fixes the shape of the functional equation
`Λ(s) = ε · conj(Λ)(1-s)`, `Λ(s) = N^{s/2} (∏ Γ-factors) L(s)`, with the **conductor N
known** and the **Γ-shifts unknown** (parameterized by a point, e.g. `(λ1, λ2)`). The
sign `ε` and the Dirichlet coefficients are also unknown. The first landscape in the
code is degree-3, conductor-1 **GL(3) Maass forms**:

```
Γ_R(s + iλ1) Γ_R(s + iλ2) Γ_R(s - i(λ1+λ2))
```

The search finds the `(λ1, λ2)` (and the `a_p`) at which a genuine L-function exists.

**Ground truth used for validation:** the first SL(3,Z) Maass form (LMFDB
`3-1-1.1-r0e3-m0.17m16.40p16.57-0`) at `(λ1, λ2) = (-16.40312474…, -0.17112189…)`, with
`ε = 1`. It is stored as `GL3_MAASS` in `afe.py` and surfaced by
`lsearch.gl3_known_target()`. **A second, independent GL(3) form was discovered during
development** at `(λ1, λ2) = (14.14163558812745, 2.380388488812225)`, `ε = 1` (see §6).

---

## 3. Mental model of the search (read this)

Per-iteration the search does a **box step** (`box_step`) and then shrinks the box
(`search`). One box step:

1. **Pick a triangular box** around the current point `P`: three corners `P`,
   `P+(h,0)`, `P+(0,h)` (`triangle_corners`).
2. **At each corner, recover the coefficients** by solving a system of equations. The
   equations come from `coefficient_relation` evaluated at the fixed symmetry point
   `s = 1/2` for a *list of weight functions* `g(s) = s^m exp(iβs + αs²)`. Each weight
   gives one (complex → two real) equation that a true L-function must satisfy. The
   unknowns are `ε = (Re, Im)` (with the constraint `|ε|² = 1`) plus two reals per prime
   `a_p`. This is `solve_at_point` (a Broyden/secant solve on a *square*, well-chosen
   subset of the equations).
3. **Detectors → zero-lines → cloud.** Equations *not* used in the solve are
   "detectors." At a true L-function every equation vanishes, so each detector's zero,
   as a function of `(λ1, λ2)`, passes through the true point. Fit each detector affinely
   over the 3 corners, take its zero-line, and intersect the lines pairwise → a **cloud**
   of candidate points (`cloud_center_spread` gives the robust median center and a
   "contains most of the points" spread).
4. **Shrink the box** to roughly the cloud spread, recenter on the cloud center, raise
   accuracy and precision (see §6), and repeat.

The cloud **center** is the current estimate of `(λ1, λ2)`; the cloud **spread** is how
well it's pinned down ("determination"). The search stops with `success` when the box
reaches `target_box`, `converged` if the box stalls (detector floor reached at the
current accuracy), or `fail`/`wandered` otherwise.

> Geometric subtlety worth internalizing: the cloud concentrates wherever the detector
> zero-lines *intersect*, which happens even where **no** L-function exists. Concentration
> alone does not prove a form (see §6, "candidate → confirm").

---

## 4. The equation generator (`coefficient_relation`)

`afe.py`'s `coefficient_relation` is a **pole-free** contour-integral identity: it drops
the `z^{-1}` term of the AFE so the contour integral is exactly 0, giving a linear
relation among the Dirichlet coefficients:

```
Σ A_n b(n)  −  ε · Σ B_n conj(b(n))  =  Σ_k r_k g(s_k)
```

(the residue sum on the right is the pole contribution; for the conductor-1 GL(3) case
there are no poles). **Dropping `z^{-1}` flips the sign of the second sum** relative to
the plain AFE — don't "fix" that sign thinking it's a bug.

`coefficient_relation_grid` computes this for a *whole list of weights at once* using one
shared Γ-function grid — a large speedup, since the Γ work dominates. The search always
calls it at `s = 1/2` (`FIXED_S`); different *weights*, not different `s`, give the many
independent equations.

The Euler product turns the `a_p` (one per prime) into the full coefficient vector
`b(1..M)`. For the tempered degree-3 family the local factor is
`1 − a_p X + conj(a_p) X² − X³`, giving the recurrence in `gl3_bppow`. Tempered ⇒ the
Satake bound **|a_p| ≤ 3** (a quick physicality check on recovered coefficients).

---

## 5. Milestones (how the code grew)

The work was staged and each stage validated against the GL(3) ground truth:

- **M0** data model: `Landscape`, `EulerProduct`, `KnownTarget`, `WeightSet`.
- **M1** Euler-product algebra: `a_p` ↔ `b(1..M)` (`gl3_bcoeffs_from_ap`, `gl3_extract_ap`).
- **M2** sizing/conditioning: `dirichlet_length` (M from accuracy), significant-prime
  selection, weight diversity.
- **M3** the solve: `solve_at_point` (Broyden, restrict-then-enlarge the prime set).
- **M4** geometry: `box_step` → detector zero-lines → candidate cloud.
- **M5/M6** the iterative shrink loop: `search`, with the precision guard and
  `converged` detection.
- **M7** resolve **many** Euler coefficients: rich weight pool + QR-selected
  well-conditioned equations (this broke the accuracy floor — see §6).

`selftest`, `selftest_m2`, `selftest_m3`, `selftest_m4` in `lsearch.py` check these.

---

## 6. Hard-won knowledge (NOT in the other docs)

**Precision rule (the one that matters).** Each cloud point's numerical error and its
genuine accuracy-limited spread are both amplified by the *same* near-parallel factor, so
that factor cancels. The correct working-precision guard is therefore **uniform**:

```
wp  ≥  accuracy + log10(cond) + 12      (GUARD_DIGITS = 12 in code)
```

A *per-point* "cloud precision ≤ box" test is wrong (it lies — it once reported 6e-16
while the cloud actually scattered over 2.8e-3). `solve_at_point` caps `cond` at
`10^(wp/2)`, which makes the guard converge to `wp ≈ 2·accuracy + 24`.

**Accuracy is coupled to box size.** λ-localization is *accuracy*-limited, not
box-limited; shrinking the box alone deadlocks. Each iteration sets
`accuracy = -log10(box) + 8`, capped at `_acc_max_for_target(target)` (= `-log10(target)
+ 10`). So a coarse target deliberately uses low accuracy (and gives coarse
coefficients); a fine target pushes accuracy up.

**The old λ2 floor and how M7 broke it.** Before M7 the search floored at ~7e-5 in λ2
because only 3 primes (a2,a3,a5) entered the solve: the weight family was too redundant
(cond 1.2e20 at just 3 primes) to add more, so the detectors bottomed out at ~1e-3. M7
fixed this with (a) a **richer weight pool** (`default_weight_set` samples a diverse
`(m, α, β)` grid; α∈{0, 1/50} is essentially free because the β=β_max weight already
fixes M) and (b) **QR row pivoting** (`_select_rows` / `_select_from_jacobian`) that
picks a *well-conditioned* square subset of equations instead of the first U in weight
order. Result: the solve climbs to ~9 primes, detector residual ~1e-13, λ error ~1e-9.

**Include the high primes even when their coefficients are garbage.** Do *not* restrict
the solve to only the cleanly-determined primes. The Dirichlet terms `n = 17, 19, 23, …`
must go somewhere; fixing those `a_p = 0` biases the well-determined low coefficients.
Letting `solve_at_point` enlarge to all significant primes (stopping only at the
conditioning cap) is correct — empirically, going from 6 to 10 free primes took the λ
determination from ±1.1e-8 to ±4.5e-17, even though `a_17…a_29` were nonsense at that
accuracy.

**Least-squares helps coefficients, NOT the λ-geometry.** `solve_at_point_lsq` fits many
more equations than unknowns (Gauss-Newton via `qr_solve`); at a known point it gives
4–33× more accurate low coefficients than the square solve. But wiring it into `box_step`
(`solver="lsq"`) does *not* reliably tighten the cloud — the detector zero-line method
needs the square solve to drive its equations *exactly* to zero, and the LSQ L2
compromise muddies that. So `box_step`/`search` default to `solver="square"` and LSQ is
used only as a **final coefficient refinement** at the converged point
(`search(refine_coeffs=True)`, reported as `result['coeff_fit_res']`). `_lsq_core`
computes its Jacobian *once* and reuses it (the equations are near-linear in the
coefficients), so a solve is ~3.7s, not minutes.

**Candidate → confirm (do not trust a single `success`).** `success` is a purely
**geometric** test (`box ≤ target`); it intentionally has *no* detector-floor gate,
because real forms found at low accuracy would be false-negatived by one. So a
low-accuracy `success` is a **candidate**. Confirm it by **resuming the search at the
found point with a finer target / higher accuracy** and checking three things:
1. the **detector residual falls** as accuracy rises (a phantom plateaus; a real form
   heads for ~1e-13);
2. the **low coefficients stabilize** and sit inside the Satake bound `|a_p| ≤ 3`;
3. `λ` and `ε` lock to stable values.
This is exactly how the second GL(3) form at (14.1416…, 2.3804…) was confirmed: a
cold-start hit it with detector residual 1.4e-4 (looked like a false positive), but
resuming drove the residual 1.4e-4 → 3.5e-6 → 7.9e-8 with stable Satake-bounded
coefficients — real after all.

**Grid search.** `wander_dist` is an **absolute** threshold (default 0.25) on how far the
cloud center may drift from the starting point. It is absolute (not box-relative) so you
can start on a grid much coarser than the box and still let the detectors pull you to a
nearby form, abandoning only genuine runaways.

---

## 7. Practical gotchas

- **High-accuracy runs are SLOW.** The bottleneck is the `afe.py` build (Γ work over the
  weight pool). A single build at accuracy ≳13 can take minutes; a full search to
  `target=1e-8` took ~40 min. Plan experiments accordingly; prefer coarse targets while
  iterating. Optimizing the `afe.py` build (or raising `solve_at_point`'s `k_init` so it
  doesn't re-enlarge from k=3 every corner) is the obvious next performance lever.
- Keep `|β|` at least ~1 below the **admissibility bound** (`admissibility_bound`,
  `(π/2)·Σκ`); right at the bound the integrand decays slowly and M explodes (e.g. 27 →
  160 terms). Keep `α ≤ 1/50`.
- macOS has **no `timeout`** command; don't use it in scripts here.
- Background jobs piped through `tail` only flush at process exit — write to a plain file
  if you want to watch progress mid-run.
- mpmath disallows `Date.now()`-style nondeterminism in some contexts; not relevant to
  the math but note all randomness/timestamps are avoided.

---

## 8. How to run

```bash
# self-tests (fast at low accuracy)
python3 lsearch.py m3        # solve recovers coefficients, rejects off-points
python3 lsearch.py m4        # box step: cloud concentrates toward the truth
python3 afe.py               # validates the analytic engine vs ζ, Dirichlet L, etc.
```

```python
import lsearch as L
from mpmath import mpf, mpc
t = L.gl3_known_target()                 # GL(3) landscape + Euler product (+ oracle)

# refine near a point (here the known first form):
res = L.search(t.landscape, t.euler,
               point=(mpf('-16.4036'), mpf('-0.1716')),
               boxsize=mpf('1e-3'), accuracy=8, working_precision=30,
               target_box=mpf('1e-6'),
               guess=t.ap, eps_guess=mpc(1))      # guess=None for a cold start
print(res['status'], res['point'], res.get('coeff_fit_res'))
```

`verbose=True` (default) prints a per-iteration report: spectral parameters with their
determination and numerical precision, the Euler coefficients, the sign, accuracy,
working precision, box size, condition number, detector residual, and primes used.

---

## 9. Where to look in `lsearch.py`

| concern | functions |
|---|---|
| data model | `Landscape`, `EulerProduct`, `KnownTarget`, `gl3_known_target` |
| Euler algebra | `gl3_bppow`, `gl3_bcoeffs_from_ap`, `gl3_extract_ap` |
| weights | `default_weight_set`, `_beta_values`, `admissibility_bound` |
| build equations | `build_equation_system`, `residual`, `coefficient_relation_grid` (in afe) |
| equation selection | `_select_rows`, `_select_from_jacobian`, `_equation_jacobian` |
| solve | `solve_at_point` (square), `solve_at_point_lsq` / `_lsq_core` (least-squares) |
| geometry | `box_step`, `triangle_corners`, `cloud_center_spread`, `estimate_cloud_precision` |
| loop | `search`, `_accuracy_for_box`, `_acc_max_for_target`, `_finalize_coeffs` |
| constants | `FIXED_S`, `ACC_OVER`, `ACC_MARGIN`, `GUARD_DIGITS` |

The git history is a good narrative: commits are milestone- and finding-shaped
(`git log --oneline`), e.g. the M7 commit, the precision-guard commit, and the
least-squares commits each explain one design decision.
