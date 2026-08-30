# -*- coding: utf-8 -*-
"""
abstention_matcher.py
=====================

Single canonical implementation of the strict and flexible abstention matchers
used throughout the ArabicaQA RAG evaluation. All evaluation scripts should
import from this module rather than redefining the patterns locally, so that
every reported abstention figure is produced by identical code.

Definitions
-----------
strict  : the normalised prediction is exactly equal to the normalised
          canonical abstention phrase prescribed by the prompt.
flexible: the normalised prediction contains (unanchored substring search)
          at least one of NO_ANSWER_PATTERNS.

By construction strict => flexible, because the canonical phrase itself
matches pattern P1. This is asserted in self_test() below.
"""

import math
import re

# ----------------------------------------------------------------------
# Canonical abstention phrase prescribed by the generation prompt
# ----------------------------------------------------------------------

NO_ANSWER = "غير موجود في السياق"


# ----------------------------------------------------------------------
# Arabic normalisation (identical to Section III-B of the manuscript)
# ----------------------------------------------------------------------

def normalize_arabic_text(s):
    """Normalise Arabic text prior to any matching or metric computation."""
    if s is None:
        return ""
    if isinstance(s, float) and math.isnan(s):
        return ""

    s = str(s).strip()

    # (i) remove diacritics
    s = re.sub(r"[\u064B-\u065F\u0670]", "", s)

    # (ii) orthographic unification
    s = re.sub(r"[إأآا]", "ا", s)
    s = re.sub(r"ى", "ي", s)
    s = re.sub(r"ؤ", "و", s)
    s = re.sub(r"ئ", "ي", s)
    s = re.sub(r"ة", "ه", s)

    # (iii) remove tatweel
    s = re.sub(r"ـ", "", s)

    # (iv) replace non-word / non-Arabic characters by a single space
    s = re.sub(r"[^\w\s\u0600-\u06FF]", " ", s)

    # (v) collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


NO_ANSWER_CANON = normalize_arabic_text(NO_ANSWER)


# ----------------------------------------------------------------------
# Flexible abstention patterns
# ----------------------------------------------------------------------
# NOTE ON ORTHOGRAPHY: patterns are matched against ALREADY NORMALISED text,
# so they are written in normalised form (ة -> ه, أ/إ/آ -> ا, ى -> ي).
# A pattern written with ة or أ can never fire and would be dead code.
#
# NOTE ON REDUNDANCY: six patterns (marked [subsumed]) are strictly implied by
# a shorter pattern earlier in the list and therefore never change a decision.
# They are retained verbatim so that this module reproduces the matcher that
# generated the published numbers; removing them leaves every flag unchanged.
# This is verified by self_test().

NO_ANSWER_PATTERNS = [
    # --- family 1: generic absence -------------------------------------
    r"غير\s+موجود",                                   # P1
    r"غير\s+مذكور",                                   # P2
    r"غير\s+متوفر",                                   # P3
    r"ليس\s+مذكور(?:ا)?",                             # P4
    r"لا\s+يوجد",                                     # P5
    r"لا\s+توجد",                                     # P6
    r"لا\s+تتوفر",                                    # P7

    # --- family 2: context-specific absence ----------------------------
    r"لا\s+يحتوي\s+السياق",                           # P8
    r"لم\s+يكن\s+(?:موجود(?:ا)?\s+)?في\s+السياق",     # P9
    r"غير\s+موجود\s+في\s+السياق",                     # P10 [subsumed by P1]
    r"غير\s+مذكور\s+في\s+السياق",                     # P11 [subsumed by P2]

    # --- family 3: information unavailable -----------------------------
    r"لا\s+توجد\s+معلومات",                           # P12 [subsumed by P6]
    r"المعلومات\s+غير\s+متوفر(?:ه|ة)",                # P13 [subsumed by P3]
    r"لا\s+يوجد\s+جواب",                              # P14 [subsumed by P5]
    r"لا\s+يوجد\s+اجابه",                             # P15 [subsumed by P5]

    # --- family 4: declared inability ----------------------------------
    r"لا\s+يمكن(?:ني)?\s+ال?اجابه",                   # P16
    r"لا\s+يمكن(?:ني)?\s+تحديد",                      # P17
    r"لا\s+يمكن(?:ني)?\s+العثور",                     # P18

    # --- family 5: explicit uncertainty --------------------------------
    r"لا\s+اعلم",                                     # P19
    r"لا\s+اعرف",                                     # P20

    # --- family 6: English fallbacks -----------------------------------
    r"not\s+found",                                   # P21
    r"not\s+mentioned",                               # P22
    r"not\s+available",                               # P23
    r"not\s+present\s+in\s+the\s+context",            # P24 [subsumed by P23? no]
    r"cannot\s+answer",                               # P25
    r"i\s+do\s+not\s+know",                           # P26
]

# Compile once. IGNORECASE is applied so that capitalised English refusals
# ("Not found in the context") are matched; Arabic is caseless, so this flag
# affects family 6 only.
_COMPILED = [re.compile(p, flags=re.IGNORECASE) for p in NO_ANSWER_PATTERNS]


# ----------------------------------------------------------------------
# Matchers
# ----------------------------------------------------------------------

def is_strict_no_answer(text):
    """Exact compliance with the prescribed abstention phrase."""
    return normalize_arabic_text(text) == NO_ANSWER_CANON


def is_flexible_no_answer(text):
    """Any recognised abstention phrase, matched as a substring."""
    t = normalize_arabic_text(text)
    if not t:
        return False
    return any(rx.search(t) for rx in _COMPILED)


# Backwards-compatible alias: earlier notebook cells used this longer name
# for the same function. Kept so that importing code needs no renaming.
is_strict_canonical_no_answer = is_strict_no_answer


def matched_patterns(text):
    """Return the indices (1-based, P-numbers) of every pattern that fires.

    Used by the validation tooling to build the error taxonomy.
    """
    t = normalize_arabic_text(text)
    if not t:
        return []
    return [i + 1 for i, rx in enumerate(_COMPILED) if rx.search(t)]


# ----------------------------------------------------------------------
# Self-tests
# ----------------------------------------------------------------------

def self_test(verbose=True):
    """Verify the structural properties asserted in the manuscript."""
    ok = True

    # 1. strict implies flexible
    assert is_strict_no_answer(NO_ANSWER)
    assert is_flexible_no_answer(NO_ANSWER)

    # 2. no dead pattern: every pattern matches at least the normalised form
    #    of its own canonical instantiation
    for i, p in enumerate(NO_ANSWER_PATTERNS):
        probe = (p.replace(r"\s+", " ")
                  .replace("(?:ا)?", "ا")
                  .replace("(?:ه|ة)", "ه")
                  .replace("(?:ني)?", "ني")
                  .replace("(?:موجود(?:ا)? )?", "موجودا ")
                  .replace("ال?", "ال"))
        if not re.search(p, normalize_arabic_text(probe), flags=re.IGNORECASE):
            ok = False
            if verbose:
                print(f"  DEAD PATTERN P{i+1}: {p!r}")

    # 3. subsumption: removing the six marked patterns changes no decision
    subsumed_idx = {10, 11, 12, 13, 14, 15}  # P-numbers
    reduced = [rx for i, rx in enumerate(_COMPILED) if (i + 1) not in subsumed_idx]
    probes = [NO_ANSWER,
              "غير موجود في السياق",
              "غير مذكور في السياق",
              "لا توجد معلومات كافية",
              "المعلومات غير متوفرة",
              "لا يوجد جواب في النص",
              "لا يوجد إجابة"]
    for s in probes:
        t = normalize_arabic_text(s)
        full = any(rx.search(t) for rx in _COMPILED)
        red = any(rx.search(t) for rx in reduced)
        if full != red:
            ok = False
            if verbose:
                print(f"  SUBSUMPTION FAILS on: {s}")

    if verbose:
        print("self_test:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    self_test()