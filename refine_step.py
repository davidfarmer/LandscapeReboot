#!/usr/bin/env python3
"""Do ONE refinement step on one entry of an L-point data file, and append the
refined entry to an output data file (format: ~/Research/RebootOfNothing/Data/FORMAT.md).

The step is one box iteration of the "L-functions out of nothing" method
(lsearch.box_step, from the directory this file is in): a triangular box with one corner at the entry's
point, the coefficients solved at each corner starting from the entry's
coefficients, the detector zero-lines intersected into a cloud, and the cloud
centre taken as the new point.  The coefficients are then solved again at the
new point (least squares over many equations), unless --no-final-solve is given.

Usage (from ~/Research/RebootOfNothing):
    python3 ~/my1/refine_step.py --data Data/R0R0R0N1.jsonl --index 0 --output Data/R0R0R0N1.jsonl

A known sign (an exact "sign" in the entry, or the landscape registry's 'sign') is held
fixed, not solved for, and written back unchanged.

Optional settings (defaults computed from the entry being refined):
    --box-size B            default: the entry's box size / 10
    --truncation-error T    default: the entry's truncation error / 10
    --working-precision W   default: max(old working precision,
                                         20 + old working precision - old detector precision),
                            so the precision never goes down, and it is kept when more than
                            20 digits remained when the old detectors were calculated
B and T may be exact rationals (24/10000000000001) or decimals (2.4e-12).
The number of Dirichlet terms is chosen so that the truncation error is below T,
that is, the accuracy is ceil(-log10 T) digits.

The accuracy of each new number is the number of digits after the decimal point
on which it agrees with the old entry (numbers the old entry does not have get -1).
The output file may be the same as the data file: the new entry is appended.
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction

UNKNOWN = -1


# ---------------------------------------------------------------------------
# Reading and writing numbers in the data format
# ---------------------------------------------------------------------------

def is_exact(x):
    return isinstance(x, str)


def text_of(x):
    """Decimal or rational text of a data-format number."""
    return x if is_exact(x) else x[0]


def to_fraction(text):
    if "/" in text:
        a, b = text.split("/")
        return Fraction(int(a), int(b))
    return Fraction(Decimal(text))


def fraction_text(q):
    return str(q.numerator) if q.denominator == 1 else "{}/{}".format(q.numerator, q.denominator)


def decimals_of(text):
    """Number of digits after the decimal point in a decimal string (exact integers: 0)."""
    if "/" in text:
        return None
    return len(text.split(".")[1]) if "." in text else 0


def fixed_text(x, decimals, mpmath):
    """mpf -> decimal string with exactly `decimals` digits after the point."""
    digits = int(mpmath.mp.dps) + 10
    s = mpmath.nstr(x, digits, strip_zeros=False, min_fixed=-mpmath.inf, max_fixed=mpmath.inf)
    with localcontext() as ctx:
        ctx.prec = digits + decimals + 10
        d = Decimal(s).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_EVEN)
        text = format(d, "f")
    if text.startswith("-") and set(text[1:]) <= set("0."):
        text = text[1:]
    return text


def agreement(new_text, old, mpmath):
    """Digits after the decimal point on which new_text agrees with the old number."""
    if old is None:
        return UNKNOWN
    old_text = text_of(old)
    diff = abs(mpmath.mpf(to_fraction(new_text).numerator) / to_fraction(new_text).denominator
               - mpmath.mpf(to_fraction(old_text).numerator) / to_fraction(old_text).denominator)
    # an exact old number (the sign "1", lambda2 = "0") is known to every decimal, so only
    # the digits written for the new number limit the agreement
    caps = [decimals_of(new_text)]
    if not is_exact(old):
        caps.append(decimals_of(old_text))
    cap = min(d for d in caps if d is not None)
    if diff == 0:
        return cap
    digits = int(mpmath.floor(-mpmath.log10(diff)))
    return max(0, min(digits, cap))


def complex_entry(z, decimals, old, mpmath):
    re_text = fixed_text(mpmath.re(z), decimals, mpmath)
    im_text = fixed_text(mpmath.im(z), decimals, mpmath)
    old_re = old_im = None
    if old is not None:
        if isinstance(old, dict):
            old_re, old_im = old["re"], old["im"]
        else:                                   # an exact real, e.g. the sign "1"
            old_re, old_im = old, "0"
    return {"re": [re_text, agreement(re_text, old_re, mpmath)],
            "im": [im_text, agreement(im_text, old_im, mpmath)]}


# ---------------------------------------------------------------------------
# The refinement step
# ---------------------------------------------------------------------------

def is_prime(n):
    return n > 1 and all(n % q for q in range(2, int(n ** 0.5) + 1))


def read_entry(path, index):
    with open(path) as handle:
        for i, line in enumerate(handle):
            if i == index:
                return json.loads(line)
    raise IndexError("{} has no entry {}".format(path, index))


def new_id(old_id):
    if ".r" in old_id:
        base, n = old_id.rsplit(".r", 1)
        if n.isdigit():
            return "{}.r{}".format(base, int(n) + 1)
    return old_id + ".r1"


def main(argv=None):
    parser = argparse.ArgumentParser(description="One refinement step for one entry of an L-point data file.")
    parser.add_argument("--data", required=True, help="data file to read the entry from")
    parser.add_argument("--index", required=True, type=int, help="entry number (line number from 0)")
    parser.add_argument("--output", required=True, help="data file to append the refined entry to")
    parser.add_argument("--box-size", help="box size (default: old box size / 10)")
    parser.add_argument("--truncation-error",
                        help="truncation error for choosing the number of terms (default: old / 10)")
    parser.add_argument("--working-precision", type=int,
                        help="working precision in digits (default: max(old working precision, "
                             "20 + old working precision - old detector precision))")
    here = os.path.dirname(os.path.abspath(__file__))
    default_my1 = here if os.path.exists(os.path.join(here, "lsearch.py")) else os.path.expanduser("~/my1")
    parser.add_argument("--my1", default=default_my1,
                        help="directory with afe.py, lsearch.py, families.py (default: the "
                             "directory of this file if lsearch.py is there, else ~/my1)")
    parser.add_argument("--landscape-file", help="landscape file instead of my1/landscapes.txt")
    parser.add_argument("--restarts", type=int, default=6,
                        help="nearby restarts if a corner solve fails (default 6)")
    parser.add_argument("--no-final-solve", action="store_true",
                        help="use the average of the corner solutions instead of solving at the new point")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    sys.path.insert(0, os.path.abspath(args.my1))
    import mpmath
    from mpmath import mp, mpf, mpc
    import lsearch
    import families

    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
    started = time.time()

    entry = read_entry(args.data, args.index)
    family = families.get_family(entry["landscape"], args.landscape_file)
    landscape, euler = family.landscape, family.euler
    if euler.complex_unknowns_per_prime != 1:
        raise NotImplementedError("only one unknown per good prime is handled so far")

    old_search = entry["search"]
    old_box = to_fraction(old_search["box_size"])
    old_trunc = to_fraction(old_search["truncation_error"])
    old_wp = float(text_of(old_search["working_precision"]))
    old_left = float(text_of(old_search["detector_precision"]))

    box = to_fraction(args.box_size) if args.box_size else old_box / 10
    trunc = to_fraction(args.truncation_error) if args.truncation_error else old_trunc / 10
    accuracy = int(math.ceil(-math.log10(trunc)))
    # never lower the precision; raise it only when fewer than 20 digits were left
    wp_rule = int(math.ceil(max(old_wp, 20 + old_wp - old_left)))
    wp = args.working_precision if args.working_precision else wp_rule
    wp_set_by_user = args.working_precision is not None
    mp.dps = wp

    point = [mpf(text_of(x)) for x in entry["point"]]
    # the unknowns are a_p at primes; stored prime powers (a_32 in the old data) follow
    # from the Euler product and are not used as starting values
    primes_old = [n for n in sorted(entry["coefficients"], key=int) if is_prime(int(n))]
    guess = {int(p): [mpc(text_of(entry["coefficients"][p]["re"]), text_of(entry["coefficients"][p]["im"]))]
             for p in primes_old}
    # The sign: an exact sign (the entry's "sign" is an exact string, or the landscape
    # registry gives it) is held fixed, not solved for, and written back unchanged.
    # Otherwise it is an unknown, started from the entry's value.
    old_sign = entry["sign"]
    if is_exact(old_sign):
        exact_sign = mpc(complex(old_sign.replace(" ", "").replace("i", "j")))
        if landscape.sign is not None and abs(landscape.sign - exact_sign) > mpf("1e-30"):
            raise ValueError("entry sign {} differs from the landscape's sign {}".format(
                old_sign, landscape.sign))
        if landscape.sign is None:
            import dataclasses
            landscape = dataclasses.replace(landscape, sign=exact_sign)
        eps_guess = exact_sign
    elif landscape.sign is not None:
        eps_guess = landscape.sign
    else:
        eps_guess = mpc(text_of(old_sign["re"]), text_of(old_sign["im"]))
    sign_known = landscape.sign is not None
    h = mpf(box.numerator) / box.denominator

    say("Refining entry {} ({}) of {}: {}".format(args.index, entry["id"], args.data, entry.get("lmfdb_label")))
    say("  point = ({})".format(", ".join(mpmath.nstr(x, 20) for x in point)))
    say("  box size {} (was {}), truncation error {} (was {}) -> accuracy {} digits".format(
        fraction_text(box), fraction_text(old_box), fraction_text(trunc), fraction_text(old_trunc), accuracy))
    say("  sign: {}".format("{} (known exactly, not an unknown)".format(old_sign if is_exact(old_sign)
                                 else landscape.sign) if sign_known else "unknown (solved for)"))
    say("  working precision {} digits (rule: max({}, 20 + {} - {:.2f}) = {}{})".format(
        wp, old_search["working_precision"], old_search["working_precision"], old_left, wp_rule,
        "; set on the command line" if wp_set_by_user else ""))

    beta_limits = lsearch._beta_limits(landscape, tuple(point), accuracy, wp)
    say("  phase limits (beta): +{} / -{}   [{:.0f} s]".format(
        mpmath.nstr(beta_limits[0], 3), mpmath.nstr(beta_limits[1], 3), time.time() - started))

    bs = None
    for attempt in range(3):
        mp.dps = wp
        step_start = time.time()
        bs = lsearch.box_step(landscape, euler, tuple(point), h, accuracy, wp,
                              guess=guess, eps_guess=eps_guess, solve_restarts=args.restarts,
                              beta_limits=beta_limits)
        if bs is None or not bs["cloud"]:
            if wp_set_by_user or attempt == 2:
                say("  box step failed at working precision {} [{:.0f} s]".format(wp, time.time() - step_start))
                bs = None
                break
            wp += 10
            say("  box step failed; raising working precision to {} and retrying".format(wp))
            continue
        need = int(math.ceil(accuracy + float(mpmath.log10(bs["cond_solve"])) + lsearch.GUARD_DIGITS))
        say("  box step done at working precision {} [{:.0f} s]: {} cloud points, solve condition {}".format(
            wp, time.time() - step_start, len(bs["cloud"]), mpmath.nstr(bs["cond_solve"], 3)))
        if wp >= need or wp_set_by_user or attempt == 2:
            if wp < need:
                say("  warning: working precision {} is below accuracy + log10(cond) + {} = {}".format(
                    wp, lsearch.GUARD_DIGITS, need))
            break
        say("  working precision {} < accuracy + log10(cond) + {} = {}; redoing at {}".format(
            wp, lsearch.GUARD_DIGITS, need, need + 3))
        wp = need + 3
    if bs is None:
        print("FAILED: no solution; nothing written.", file=sys.stderr)
        return 1

    center, spread = lsearch.cloud_center_spread(bs["cloud"])
    cloud_prec = lsearch.estimate_cloud_precision(bs)
    det_res = max((s["det_res"] for s in bs["sols"] if s), default=mpf(0))
    sol = lsearch._average_solution(bs["sols"])
    fit_res = None
    num_terms = None
    if not args.no_final_solve:
        final_start = time.time()
        sysf = lsearch.build_equation_system(landscape, euler, tuple(center), accuracy, wp,
                                             beta_limits=beta_limits)
        num_terms = sysf.M
        ls = lsearch.solve_at_point_lsq(sysf, guess=sol["ap"], eps_guess=sol["epsilon"])
        if ls is not None:
            sol = {"primes": ls["primes"], "ap": ls["ap"], "epsilon": ls["epsilon"]}
            fit_res = ls["all_res"]
        say("  final least-squares solve at the new point [{:.0f} s]: fit residual {}".format(
            time.time() - final_start, mpmath.nstr(fit_res, 3) if fit_res is not None else "failed"))

    # digits of precision remaining when the detectors were calculated
    left = wp - float(mpmath.log10(bs["cond_solve"]))
    decimals = max(1, int(math.floor(left)))

    new_point = []
    for x, old in zip(center, entry["point"]):
        t = fixed_text(x, decimals, mpmath)
        new_point.append([t, agreement(t, old, mpmath)])
    coefficients = {}
    for p in sol["primes"]:
        coefficients[str(p)] = complex_entry(sol["ap"][p][0], decimals, entry["coefficients"].get(str(p)), mpmath)
    if sign_known:
        if is_exact(old_sign):
            sign = old_sign
        elif mpmath.im(landscape.sign) == 0 and mpmath.re(landscape.sign) in (1, -1):
            sign = str(int(mpmath.re(landscape.sign)))
        else:                              # a known non-real sign from the registry
            sign = {"re": [mpmath.nstr(mpmath.re(landscape.sign), 30), UNKNOWN],
                    "im": [mpmath.nstr(mpmath.im(landscape.sign), 30), UNKNOWN]}
    else:
        sign = complex_entry(sol["epsilon"], decimals, old_sign, mpmath)

    elapsed = time.time() - started
    refined = {
        "id": new_id(entry["id"]),
        "landscape": entry["landscape"],
        "lmfdb_label": entry.get("lmfdb_label"),
        "lmfdb_dual_label": entry.get("lmfdb_dual_label"),
        "self_dual": entry.get("self_dual"),
        "point": new_point,
        "sign": sign,
        "coefficients": coefficients,
        "search": {
            "box_size": fraction_text(box),
            "truncation_error": fraction_text(trunc),
            "working_precision": str(wp),
            "detector_precision": ["{:.4f}".format(left), UNKNOWN],
            "accuracy_digits": str(accuracy),
            "num_terms": str(num_terms) if num_terms else None,
            "determination": mpmath.nstr(spread, 3),
            "cloud_point_precision": mpmath.nstr(cloud_prec, 3) if cloud_prec is not None else None,
            "detector_residual": mpmath.nstr(det_res, 3),
            "coefficient_fit_residual": mpmath.nstr(fit_res, 3) if fit_res is not None else None,
            "cloud_points": len(bs["cloud"]),
        },
        "source": entry.get("source"),
        "history": list(entry.get("history", [])) + [{
            "parent": entry["id"],
            "parent_file": os.path.basename(args.data),
            "parent_index": args.index,
            "program": "refine_step.py",
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "seconds": round(elapsed, 1),
            "command": " ".join(sys.argv),
        }],
    }
    with open(args.output, "a") as handle:
        handle.write(json.dumps(refined, separators=(",", ":")) + "\n")

    say("")
    say("New point (digits agreeing with the old point after the decimal point):")
    for i, (t, a) in enumerate(new_point, 1):
        say("  lambda{} = {}...  agreement {}".format(i, t[:t.index(".") + min(a + 6, decimals) + 1], a))
    say("  determination +- {}, cloud precision {}, detector residual {}".format(
        mpmath.nstr(spread, 3), mpmath.nstr(cloud_prec, 3) if cloud_prec is not None else "?",
        mpmath.nstr(det_res, 3)))
    agree = [coefficients[p][part][1] for p in coefficients for part in ("re", "im")
             if coefficients[p][part][1] != UNKNOWN]
    say("  coefficients: {} primes; agreement with old: min {} / max {} digits".format(
        len(coefficients), min(agree) if agree else "-", max(agree) if agree else "-"))
    if sign_known:
        say("  sign = {} (known exactly; not solved for)".format(sign))
    else:
        say("  sign = {} + {} i (agreement {}, {})".format(
            mpmath.nstr(mpmath.re(sol["epsilon"]), 15), mpmath.nstr(mpmath.im(sol["epsilon"]), 3),
            sign["re"][1], sign["im"][1]))
    say("Appended entry {} to {} [{:.0f} s total]".format(refined["id"], args.output, elapsed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
