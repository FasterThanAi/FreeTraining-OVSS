"""Two papers, one set of numbers — enforce it.

`main.tex` has no length target; `main_cvpr.tex` is cut to a conference limit. They
share `numbers.tex` and `refs.bib`, which is the whole safety argument: a value can
be changed once and both papers change, or neither does.

⛔ The failure this prevents is specific and has bitten this project five times in
other forms: a number that is right in one document and stale in the other. The rule
in `numbers.tex` — *if it is not a macro here, it has no citation and does not belong*
— only holds if nothing is typed inline. So this checks exactly that, plus the
cross-document consistency the second file makes possible.

Run before every bundle. Exits non-zero on any failure.
"""
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / 'paper'
DOCS = ['main.tex', 'main_cvpr.tex', 'supplementary.tex']

# Inline-value debt on the day this check was added. main_cvpr.tex is new and is
# held to zero. The other two may only improve.
BACKLOG = {'main.tex': 70, 'supplementary.tex': 23, 'main_cvpr.tex': 0}

# Numerals that are structural, not results: font sizes, margins, class options,
# section/figure numbering, and the two-column layout itself.
SKIP_LINE = re.compile(r'\\(documentclass|usepackage|geometry|includegraphics|label|ref|cite|'
                       r'newcommand|renewcommand|input|bibliography|graphicspath|itemsep|'
                       r'begin\{tabular\}|hspace|vspace|setlength|columnwidth|textwidth)')
# A bare decimal or a signed figure in prose is what we are hunting.
INLINE = re.compile(r'(?<![\w\\{.])[+-]?\d+\.\d+(?![\w}])')


def main():
    fail = 0
    nums = (PAPER / 'numbers.tex').read_text()
    defined = dict(re.findall(r'\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}', nums))
    print(f'numbers.tex defines {len(defined)} macros')

    # -- 0. a macro defined TWICE is a LaTeX error, not a style problem --------
    # ⛔ `\newcommand` refuses to redefine, so a duplicate name halts the build.
    # This happened twice in one session: a macro was added without checking the
    # name existed. A compiler catches it in seconds -- but only if a compiler is
    # available, and on the authoring machine there is none.
    names = re.findall(r'\\newcommand\{\\([a-zA-Z]+)\}', nums)
    dup = sorted({x for x in names if names.count(x) > 1})
    if dup:
        print(f'  FAIL  defined more than once (LaTeX will halt): {dup}')
        fail += 1
    else:
        print('  ok    no macro defined twice')

    # -- 0b. non-ASCII in body text is a LaTeX error under inputenc ------------
    # ⛔ This project writes its notes with ⚠/⛔/⭐ markers and they leaked into
    # main.tex on 8 and 13 Sep, breaking every build since. They were invisible
    # because the build check grepped for undefined REFERENCES, not for errors.
    # Comments are safe; anything before an unescaped % is not.
    SAFE = set('\u2010\u2011\u2013\u2014\u2018\u2019\u201c\u201d\u2026')
    for doc in DOCS + ['numbers.tex']:
        f = PAPER / doc
        if not f.exists():
            continue
        hits = []
        for i, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split('%')[0]
            bad = [c for c in code if ord(c) > 0x2000 and c not in SAFE]
            if bad:
                hits.append((i, ''.join(bad), code.strip()[:60]))
        if hits:
            print(f'  FAIL  {doc}: {len(hits)} line(s) with non-ASCII in body text:')
            for i, b, ctx in hits[:6]:
                print(f'       line {i:>5}  {b!r}  {ctx}')
            fail += 1
    if not fail:
        print('  ok    no non-ASCII outside comments')
    print()

    for name in DOCS:
        f = PAPER / name
        if not f.exists():
            print(f'FAIL  {name} is missing')
            fail += 1
            continue
        s = f.read_text()

        # -- 1. every macro used must be defined ------------------------------
        local = set(re.findall(r'\\newcommand\{\\([a-zA-Z]+)\}', s))
        used = set(re.findall(r'\\([a-zA-Z]+)\{\}', s))
        miss = sorted(used - set(defined) - local - {'date'})
        print(f'{name}: {len(s.split()):>6} words, {len(used & set(defined))} macros used')
        if miss:
            print(f'  FAIL  undefined: {miss}')
            fail += 1

        # -- 2. no result typed inline ---------------------------------------
        bad = []
        for i, line in enumerate(s.splitlines(), 1):
            if line.lstrip().startswith('%') or SKIP_LINE.search(line):
                continue
            for m in INLINE.findall(line):
                bad.append((i, m, line.strip()[:70]))
        # ⚠️ A RATCHET, not a clean-room rule. main.tex and supplementary.tex
        # carry inline values that predate this check; fixing all of them at once
        # would be a large mechanical edit with its own risk of transcription
        # error. So the budget below is the count on the day the check was added,
        # and the build fails if it RISES. New documents get a budget of zero.
        budget = BACKLOG.get(name, 0)
        if len(bad) > budget:
            print(f'  FAIL  {len(bad)} inline decimal(s), budget {budget} — '
                  f'each new one needs a macro in numbers.tex:')
            for i, m, ctx in bad[:12]:
                print(f'       line {i:>5}  {m:>8}   {ctx}')
            fail += 1
        elif bad:
            print(f'  ok    {len(bad)} inline decimal(s), within the known '
                  f'backlog of {budget} (ratchet: this may fall, never rise)')
        else:
            print('  ok    no value typed inline')

        # -- 3. environments balanced ----------------------------------------
        for env in ('table', 'tabular', 'figure', 'figure*', 'abstract', 'document'):
            b, e = s.count('\\begin{%s}' % env), s.count('\\end{%s}' % env)
            if b != e:
                print(f'  FAIL  {env}: {b} begin vs {e} end')
                fail += 1
        if s.count('{') != s.count('}'):
            print(f'  FAIL  braces unbalanced ({s.count("{")} vs {s.count("}")})')
            fail += 1

    # -- 4. the two papers must not disagree about a shared claim -------------
    # They share macros, so values cannot differ. What CAN differ is a claim
    # stated in one and contradicted in the other, which no script can catch.
    # What it can catch is a macro the short paper redefines locally.
    short = (PAPER / 'main_cvpr.tex').read_text()
    full = (PAPER / 'main.tex').read_text()
    for doc, s in (('main.tex', full), ('main_cvpr.tex', short)):
        shadow = [m for m in re.findall(r'\\newcommand\{\\([a-zA-Z]+)\}', s)
                  if m in defined]
        if shadow:
            print(f'\nFAIL  {doc} redefines shared macro(s): {shadow}')
            print('      That breaks the one-source-of-truth guarantee.')
            fail += 1

    # -- 5. informational: values shared by more than one macro ---------------
    # ⭐ NOT an error. Distinct quantities legitimately share a value. This is
    # printed because it is the evidence for the one rule that keeps this file
    # honest: NEVER choose a macro by grepping for its value. Three real near
    # misses while writing the conference version -- 54.7 is both UAVid's
    # published mIoU and LoveDA water's recall; +0.16 is both OpenEarthMap's
    # threshold gain and lever four's null; +0.886 is both a label-free proxy's
    # correlation and lever four's presence ordering.
    import collections
    by_val = collections.defaultdict(list)
    for name, val in re.findall(r'\\newcommand\{\\([a-zA-Z]+)\}\{([^}]*)\}', nums):
        by_val[val].append(name)
    shared = {v: n for v, n in by_val.items() if len(n) > 1}
    worst = sorted(shared.items(), key=lambda kv: -len(kv[1]))[:5]
    print(f'\n{len(shared)} values are used by more than one macro '
          f'(not an error -- the reason never to match a macro by its value):')
    for v, names in worst:
        print(f'   {v:>10}  ->  {", ".join(names)}')

    print('\n' + ('ALL PASS' if not fail else f'{fail} FAILURE(S)'))
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
