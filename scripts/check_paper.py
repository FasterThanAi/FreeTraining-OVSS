"""
Pre-flight structural check for the paper, before a first LaTeX build.

Catches the things that make a first compile fail or, worse, succeed with a
silent `??`: a \\ref with no \\label, a \\cite key absent from refs.bib, an
unbalanced environment, and a tabular row whose cell count does not match its
column spec.

⚠️ It also reports labels that are never referenced. That is not an error, but it
found a real one: the paper's central figure was labelled and never pointed at,
so no reader was ever told to look at it.

⚠️ \\multicolumn spans several columns by design, so a row containing one is
counted by its span rather than its ampersands. An earlier version flagged every
such row and would have trained its reader to ignore it.

    python scripts/check_paper.py
"""
import re
import sys
from pathlib import Path


def cells(row, ncol):
    """Cell count for one tabular row, honouring \\multicolumn spans."""
    spans = [int(n) for n in re.findall(r'\\multicolumn\{(\d+)\}', row)]
    stripped = re.sub(r'\\multicolumn\{\d+\}\{[^}]*\}\{(?:[^{}]|\{[^{}]*\})*\}', '', row)
    return stripped.count('&') + 1 + sum(s - 1 for s in spans) if spans else row.count('&') + 1


def main():
    root = Path(__file__).resolve().parents[1] / 'paper'
    bib = set(re.findall(r'@\w+\{([^,]+),', (root / 'refs.bib').read_text()))
    ok = True

    for name in ('main.tex', 'supplementary.tex'):
        f = root / name
        if not f.exists():
            continue
        s = f.read_text()
        src = '\n'.join('' if l.lstrip().startswith('%') else l.split('%')[0]
                        for l in s.split('\n'))
        print(f'\n== {name} ==')

        labels = set(re.findall(r'\\label\{([^}]+)\}', src))
        refs = set(re.findall(r'\\(?:ref|Cref|cref|autoref)\{([^}]+)\}', src))
        missing = sorted(refs - labels)
        print(f'  {len(labels)} labels, {len(refs)} refs | undefined: {missing or "none"}')
        ok &= not missing
        never = sorted(x for x in labels - refs if x.startswith(('fig:', 'tab:')))
        if never:
            print(f'  ⚠ float labelled but NEVER referenced: {never}')
            print('    (not fatal, but a float no sentence points at is a float nobody reads)')

        cites = set()
        for c in re.findall(r'\\cite[tp]?\{([^}]+)\}', src):
            cites |= {x.strip() for x in c.split(',')}
        bad = sorted(cites - bib)
        print(f'  {len(cites)} citations | not in refs.bib: {bad or "none"}')
        ok &= not bad

        envs = {}
        for kind, e in re.findall(r'\\(begin|end)\{(\w+\*?)\}', src):
            envs[e] = envs.get(e, 0) + (1 if kind == 'begin' else -1)
        unb = {k: v for k, v in envs.items() if v}
        print(f'  environments | unbalanced: {unb or "none"}')
        ok &= not unb

        for m in re.finditer(r'\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}',
                             src, re.S):
            ncol = len(re.sub(r'[^lcrp]', '', re.sub(r'p\{[^}]*\}', 'p', m.group(1))))
            for row in m.group(2).split('\\\\'):
                r = re.sub(r'\\(cmidrule|midrule|toprule|bottomrule)\S*', '', row).strip()
                if not r or r.startswith('\\'):
                    continue
                n = cells(r, ncol)
                if n != ncol:
                    print(f'  ⚠ tabular({ncol} cols): row has {n} | {r[:60]}')
                    ok = False

    print('\n' + ('OK -- no structural problems found' if ok
                  else 'FIX the flagged items before building'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
