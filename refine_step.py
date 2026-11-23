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

If the box step fails, it is retried with 10 more digits of working precision, up to
--max-retries times (default 6), also when --working-precision was given.  The phase limits
(the largest usable |beta| of the weight functions) are probed again whenever the working
precision changes: probed at a low precision they stop short, which gives poorer weights.
The triangle of test points is rotated about the point: the first attempt by --start-angle
(default 'random'; 'carry' continues from the entry being refined; or a number), and each
new attempt by --box-angle-step more (default the golden angle, as lsearch.search does when
zooming in).  The angle of each attempt is recorded, so a run can be repeated exactly by
giving its first angle as --start-angle.
Each attempt (search.attempts) records its working precision, phase limits, angle, and result,
including which step failed.  If every attempt
fails, a failure record is appended to the output file instead of a refined entry: a line
with "status": "failed", a "message", and the working precisions tried (search.attempts).
It has no point or coefficients; refine_step.py refuses to refine it, and convert_new.py
skips it.
"""

import argparse
import datetime
import json
import math
import os
import random
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
                entry = json.loads(line)
                if entry.get("status") == "failed":
                    parent = entry["history"][-1]
                    raise SystemExit("entry {} of {} is the record of a failed refinement ({}); refine "
                                     "its parent instead: entry {} of {}".format(
                                         index, path, entry["id"], parent.get("parent_index"),
                                         parent.get("parent_file")))
                return entry
    raise IndexError("{} has no entry {}".format(path, index))


WP_STEP = 10      # digits added to the working precision for each retry
BOX_ANGLE_STEP = "2.39996"   # radians; the golden angle, as in lsearch.search


def start_angle(entry, spec, step, mpmath):
    """The rotation of the first attempt's test points (radians), from --start-angle:
    'random' (uniform in [0, 2 pi)), 'carry' (the last angle recorded in the entry's
    search.attempts plus one step; 0 if it has none, as for the Mathematica entries), or a
    number.  Returns (angle, how it was chosen)."""
    spec = str(spec).strip().lower()
    if spec == "random":
        return mpmath.mpf(random.random()) * 2 * mpmath.pi, "random"
    if spec == "carry":
        attempts = (entry.get("search") or {}).get("attempts") or []
        angles = [a["angle"] for a in attempts if a.get("angle") not in (None, "")]
        if not angles:
            return mpmath.mpf(0), "carried from the parent (no angle recorded there: 0)"
        return mpmath.mpf(angles[-1]) + step, "carried from the parent ({} + one step)".format(angles[-1])
    return mpmath.mpf(spec), "given"


def failure_record(entry, args, box, trunc, accuracy, attempts, elapsed):
    """The line written to the output file when every attempt failed: no point or
    coefficients, "status": "failed", and what was tried."""
    return {
        "id": new_id(entry["id"]) + "-failed",
        "status": "failed",
        "landscape": entry["landscape"],
        "lmfdb_label": entry.get("lmfdb_label"),
        "lmfdb_dual_label": entry.get("lmfdb_dual_label"),
        "message": "refine_step.py: the box step failed at every working precision tried ({}); "
                   "nothing was computed.  Try a larger --working-precision or --max-retries, or a "
                   "different --box-size.".format(", ".join(a["working_precision"] for a in attempts)),
        "search": {
            "box_size": fraction_text(box),
            "truncation_error": fraction_text(trunc),
            "accuracy_digits": str(accuracy),
            "attempts": attempts,
        },
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
    parser.add_argument("--max-retries", type=int, default=6,
                        help="if the box step fails, retry this many times, each time with %d more "
                             "digits of working precision (default 6)" % WP_STEP)
    parser.add_argument("--start-angle", default="random",
                        help="rotation (radians, about the point) of the first attempt's test points: "
                             "'random' (default), 'carry' (continue from the entry being refined: its "
                             "last recorded angle plus one --box-angle-step), or a number")
    parser.add_argument("--box-angle-step", default=BOX_ANGLE_STEP,
                        help="rotate the triangle of test points by this further angle (radians) for "
                             "each new attempt at a higher precision, as lsearch.search does for each "
                             "zoom iteration (default %s, the golden angle; 0: the same angle for every "
                             "attempt)" % BOX_ANGLE_STEP)
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
        "; set on the command line" if args.working_precision is not None else ""))

    def probe_phase_limits(wp):
        """The largest usable |beta| of the weight functions, probed at this working
        precision: at a low precision the probe stops short, and the weights are poorer."""
        t = time.time()
        mp.dps = wp
        limits = lsearch._beta_limits(landscape, tuple(point), accuracy, wp)
        say("  phase limits (beta) at working precision {}: +{} / -{}   [{:.0f} s]".format(
            wp, mpmath.nstr(limits[0], 4), mpmath.nstr(limits[1], 4), time.time() - t))
        return limits

    beta_limits = probe_phase_limits(wp)
    probed_at = wp

    # The entry is a known L-function, so a failed box step is retried at higher working
    # precision (WP_STEP more digits each time, at most --max-retries times).  A step whose
    # working precision is below lsearch's guard (accuracy + log10(cond) + GUARD_DIGITS) is
    # redone once at the guard; that redo is not counted as a retry.
    attempts = []
    bs = None
    retries = 0
    guard_redone = False
    angle_step = mpf(args.box_angle_step)
    angle0, how = start_angle(entry, args.start_angle, angle_step, mpmath)
    angle0 = angle0 % (2 * mpmath.pi)
    say("  test points rotated by {} radians about the point ({}); {} more for each new attempt".format(
        mpmath.nstr(angle0, 6), how, mpmath.nstr(angle_step, 6)))
    while True:
        # attempt n: the triangle rotated by angle0 + n steps
        angle = (angle0 + angle_step * len(attempts)) % (2 * mpmath.pi)
        angle_text = mpmath.nstr(angle, 6) if angle else "0"
        if wp != probed_at:                  # the phase limits depend on the precision
            beta_limits = probe_phase_limits(wp)
            probed_at = wp
        limits_text = [mpmath.nstr(beta_limits[0], 4), mpmath.nstr(beta_limits[1], 4)]
        mp.dps = wp
        step_start = time.time()
        if attempts:
            say("  test points rotated by {} radians about the point".format(angle_text))
        bs = lsearch.box_step(landscape, euler, tuple(point), h, accuracy, wp,
                              guess=guess, eps_guess=eps_guess, solve_restarts=args.restarts,
                              beta_limits=beta_limits, angle=angle)
        seconds = round(time.time() - step_start, 1)
        if bs is None or not bs["cloud"]:
            if bs is None:
                reason = lsearch.LAST_BOX_STEP_FAILURE or "box step failed (no reason recorded)"
            else:
                reason = "no cloud points: the {} detector lines gave no usable intersections".format(
                    len(bs.get("lines") or []))
            attempts.append({"working_precision": str(wp), "phase_limits": limits_text, "angle": angle_text,
                             "result": "failed: " + reason, "seconds": seconds})
            bs = None
            say("  box step failed at working precision {} [{:.0f} s]: {}".format(wp, seconds, reason))
            if retries >= args.max_retries:
                say("  no retries left")
                break
            retries += 1
            wp += WP_STEP
            say("  raising working precision to {} and retrying (retry {} of {})".format(
                wp, retries, args.max_retries))
            continue
        need = int(math.ceil(accuracy + float(mpmath.log10(bs["cond_solve"])) + lsearch.GUARD_DIGITS))
        say("  box step done at working precision {} [{:.0f} s]: {} cloud points, solve condition {}".format(
            wp, seconds, len(bs["cloud"]), mpmath.nstr(bs["cond_solve"], 3)))
        if wp >= need or guard_redone:
            attempts.append({"working_precision": str(wp), "phase_limits": limits_text, "angle": angle_text,
                             "result": "ok", "seconds": seconds})
            if wp < need:
                say("  warning: working precision {} is below accuracy + log10(cond) + {} = {}".format(
                    wp, lsearch.GUARD_DIGITS, need))
            break
        attempts.append({"working_precision": str(wp), "phase_limits": limits_text, "angle": angle_text,
                         "result": "below the guard {}; redone".format(need), "seconds": seconds})
        say("  working precision {} < accuracy + log10(cond) + {} = {}; redoing at {}".format(
            wp, lsearch.GUARD_DIGITS, need, need + 3))
        wp = need + 3
        guard_redone = True
    if bs is None:
        record = failure_record(entry, args, box, trunc, accuracy, attempts, time.time() - started)
        with open(args.output, "a") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        print("FAILED: the box step failed at working precision {}; a failure record {} was "
              "appended to {}.".format(", ".join(a["working_precision"] for a in attempts),
                                       record["id"], args.output), file=sys.stderr)
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
            "attempts": attempts,
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
