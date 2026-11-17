# Originality check

Run on the compiled submission PDF (`INDIACom2027_Track2_paper_named.pdf`, 6 pages, 5,772 words) with
`scratchpad/v2/selfcheck.py`. This is **not** a substitute for Turnitin or iThenticate; INDIACom runs Turnitin in
its first review round, and the authors should run their own check before submitting. What this records is the one
risk specific to this project: the manuscript describes work whose text and data are in a public repository, so a
similarity checker that indexes that repository could report matches against it.

## Self-overlap with the public repository

Every Markdown file published in the repository was compared against the paper text, looking for runs of eight or
more consecutive words in common (the unit a similarity checker highlights).

| File | Matching runs | Nature of the match |
|---|---|---|
| `paper/REFERENCES.md` | 4 | Titles of cited works, e.g. "a review of proton exchange membrane water electrolysis on degradation mechanisms and mitigation strategies, Journal of Power Sources" |
| `audit/01_research_audit.md` | 2 | "rounding of the action and of the observation"; "hydrogen, degradation and reward agree with relative difference 0" |
| `audit/02_track2_plan.md` | 1 | The weight list `w2 in {1.41, 5.32, 12.9, 20}` |
| `kaggle/README.md` | 1 | The ratio and weight `1/8 and 1/4 at w2 = 20` |
| `results_batch/README.md` | 1 | A software version string, "Linux, Python 3.13, NumPy 2.1.3" |

**Share of the paper inside any repeated run: 1.7%.** Every match is a citation title, a parameter list, a
software version, or a short technical phrase with no natural paraphrase. No passage of prose is shared. A checker
reporting these would be reporting bibliography and numbers, which reviewers discount.

## Originality of the text

The manuscript was written for this submission. The background sentences that are most likely to resemble standard
phrasing (the definitions of the Karp--Flatt serial fraction, Hockney's model, Amdahl's and Gustafson's laws, and
the update-to-data ratio) were checked individually: each states the concept in our own words and cites its source,
rather than reproducing a definition verbatim. No text was copied from any cited work.

## What the authors should still do

1. Run the PDF through Turnitin or iThenticate before submission, as the INDIACom guidelines advise, and keep the
   originality report; the site states it may be submitted with the paper.
2. If a report flags the repository, the overlap table above explains it: the repository is the authors' own
   artifact for this paper, not a third-party source.
3. The repository link is withheld from the submitted PDF for double-blind review, so an indexed match cannot
   identify the authors from the paper itself.
