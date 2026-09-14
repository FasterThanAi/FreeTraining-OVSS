# Pre-registration — is lever 2's UAVid gain repairing the VOCABULARY?

**Committed 14 Sep 2026, before any cache with the new prompt exists.** `git log` is the
timestamp. Scored in `UAVID_RESULTS.md`; nothing here is edited after the run.

## The suspicion, stated plainly

Lever 2 adds **+5.64** on UAVid, and **96%** of it is two classes: `tree` **+20.79** and
`vegetation` **+13.12**, separated by a **6.4×** scale ratio (2.522 vs 0.383).

⚠️ **UAVid's class is *low vegetation*. The prompt is the bare word `vegetation`, which
plainly also names trees.** A large reweighting between two prompts that do not separate is
exactly what a bad prompt pair looks like from the inside. `PROMPT_ENSEMBLE_RESULTS` measured
the vocabulary as worth **+4.94 mIoU** on LoveDA — larger than every calibration lever
combined — so this is not a remote possibility.

**If lever 2 is mostly repairing a word, the paper must say so.** A reviewer will think of
this, and it is far better to answer it than to be asked.

## The intervention — one variable

`cls_uavid.txt` line 6: **`vegetation` → `low vegetation`**. Nothing else changes.

✅ **No arity confound.** `WEEK3 §7b` and `PROMPT_ENSEMBLE_RESULTS` both record that prompt
*count* moves scores regardless of semantics (`max`-of-3 beats `max`-of-1). A two-word phrase
on one line is still **one** prompt, so arity stays at 1 for every class. ⛔ No commas.

## Predictions

**V1.** The **baseline** mIoU (published τ, no scale) **rises** above 56.86. *Reason:* the
prompt then names the class the dataset actually annotates.

**V2 ⭐ — the diagnostic.** At the baseline, `vegetation`'s **precision rises** (from 53.62)
and `tree`'s **recall rises** (from 55.90). Those two numbers are the confusion the prompt is
supposed to fix, and they should move before mIoU does.

**V3 ⭐⭐ — THE TEST.** Lever 2's transfer gain **falls below +2.8**, i.e. loses more than
half of its +5.64, *if* the gain was mostly prompt repair.
⚠️ Stated as a threshold, not a direction, so it cannot be read as confirmed either way after
the fact.

**V4.** The fitted `tree`/`vegetation` scale ratio **falls below 3.0** (from 6.4×). *Reason:*
if the prompts separate the classes, the argmax needs less help to.

**V5.** Lever 1's gain stays within **±0.5** of **+1.04**. *Reason:* thresholds act on
whether a class clears a bar, not on which class wins — a different error mass.

## Branch table — written before the number

| outcome | what it means |
|---|---|
| **V1 and V3 both hold** | ⛔ **Lever 2 was substantially repairing the vocabulary on this dataset.** Report it. The lever still works on LoveDA, Potsdam and ConInfer, where no such prompt defect is known — but UAVid's headline must be restated with the corrected prompt. |
| **V1 holds, V3 fails** | ⭐ **The best outcome.** A better prompt helps *and* lever 2 still helps by as much: the two are independent, the argmax competition is a genuine model limitation, and the paper gains a vocabulary ablation it did not have. |
| **V1 fails, V3 fails** | the prompt was never the problem; lever 2 addresses a real limitation. Report V1 as a refuted suspicion of my own. |
| **V1 fails, V3 holds** | incoherent — a prompt that does not help the baseline should not remove lever 2's headroom. Treat as a fault in the run and check the cache before reading anything. |

⚠️ **Whatever happens, the corrected prompt becomes the reported configuration if V1 holds.**
Reporting a number obtained with a prompt we know to be wrong, because it flatters the
method, is the error this pre-registration exists to prevent.

⛔ **One arm only.** `grass` and `low vegetation,grass` are both tempting and both change more
than one thing at a time (the second changes arity). If this arm is ambiguous, a second is
cheap — but it gets its own pre-registration.
