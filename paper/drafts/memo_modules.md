# What the modules contribute: evidence summary

One page for the supervisor discussion. Every number below comes from
`runs/*/cv_eval.json` and can be reproduced with
`.venv/bin/python paper/drafts/make_tables.py`.

## Why the thesis number (+5.75 points) cannot be reproduced

The thesis compared X3D-M with MA-X3D at 90.5% and reported the epoch with the best
**test** accuracy, on the official RWF-2000 split. Two things inflate that comparison:

1. **Selection on the test split.** We now log test accuracy after every epoch without
   using it. Choosing the best test epoch instead of the epoch chosen on a validation
   fold raises the reported accuracy by **+0.38 to +1.38 points** (mean +0.81, max over
   folds +1.75), measured over six models.
2. **Duplicated clips.** Nine training clips of the official release are exact copies of
   test clips, and nine more pairs are duplicated inside the training split. A model
   trained on the release has already seen part of its test set.

Neither of these is specific to our work; they affect most published numbers on this
benchmark, which is why the paper reports them.

## The modules, measured cleanly

Protocol: duplicates removed, grouped 4-fold cross-validation, decisions on out-of-fold
(OOF) predictions of 1,583 clips, test split used once, paired McNemar test.

| Dataset | Recipe | X3D-M | MA-X3D | Difference (OOF) | p |
| --- | --- | --- | --- | --- | --- |
| RWF-2000 | conventional (thesis-like) | 88.00 | 88.99 | +0.44 [-0.76, +1.58] | 0.52 |
| RWF-2000 | tuned (final) | 90.90 | 90.86 | -0.04 [-0.72, +0.63] | 0.78 (3 seeds) |
| Hockey Fights | conventional | 94.73 | **95.98** | **+1.25 [+0.50, +2.13]** | **0.006** |
| Hockey Fights | tuned (final) | 96.11 | 96.61 | +0.50 [-0.25, +1.25] | 0.34 |

Supporting measurements on RWF-2000 (final recipe, one seed unless noted):

- Motion Attention alone: 90.65 (-0.19 vs X3D-M, p = 0.75).
- Wide kernel alone: 91.03 (+0.19, p = 0.81).
- **Control**: the same gate with an all-zero motion map reaches 91.09, i.e. as good as
  with the real motion map. On this dataset the gate's behaviour does not come from the
  motion cue.
- The trained network does use the ring: zeroing it at test time drops accuracy from
  87.6% to 80.6%, and the ring holds 30% of the kernel magnitude. But training without
  it (plain X3D-M) reaches the same accuracy, so the network does not need it.
- MA-X3D is ahead of X3D-M only on the quartile of clips with the strongest motion
  (+1.26 points); the three quieter quartiles show no difference.

## What does explain the improvement

| Change (MA-X3D, RWF-2000) | OOF | vs previous row | p |
| --- | --- | --- | --- |
| conventional recipe | 88.99 | - | - |
| backbone lr 1e-5 -> 5e-5 | 89.70 | +1.26 | 0.040 |
| + 48 epochs, no teacher | 90.46 | +0.76 | - |
| + distillation (48 epochs) | 90.86 | +0.57 vs the row above | 0.37 |

The learning rate and the longer schedule account for the gain. Distillation from the
86M-parameter VideoMAE-B teacher adds 0.57 points on top of the same schedule without a
teacher, which is not significant at this sample size.

## What we can claim in the paper

- The modules give a **significant gain on Hockey Fights** under the conventional recipe
  (+1.25, p = 0.006) and on RWF-2000 they improve the test split under recipe R1 by
  +1.5 ± 0.8 points in the fold-paired comparison.
- They cost +1.2% parameters and +15% FLOPs, fuse away at inference and never hurt.
- On RWF-2000 with the tuned recipe the gain is within noise, and we say so.

## What we cannot claim

That the modules improve accuracy by several points, or that they are the reason for the
final numbers. A reviewer can retrain an X3D-M baseline in a few GPU-hours; our own
X3D-M baseline under the same recipe reaches 90.90 out-of-fold and 87.4% test accuracy,
against 90.86 and 87.5% for MA-X3D.

## If we want a stronger module result

Options, in order of cost:

1. More seeds for the +MA and +WK rows (running): a small true effect may reach
   significance with 3 seeds x 4 folds.
2. Test the modules where they showed an effect: the conventional recipe and Hockey, with
   3 seeds, and report that regime explicitly.
3. Change the module so that it addresses what limits the tuned baseline, e.g. a stronger
   motion pathway rather than a bounded gate, and re-measure. This is new research, not
   rewriting.
