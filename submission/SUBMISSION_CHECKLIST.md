# KANABI — SIH 2026 idea submission checklist (PS SIH26168)

Deadline: 5 Oct 2026 (tonight). Work through this top to bottom.

## 1. The deck (must-do)

- [ ] Open `KANABI_SIH26168_IdeaPPT.pptx` (built from the official SIH template, 6 slides).
- [ ] Slide 1: replace `<Team ID from portal>` with the real Team ID from the SIH portal / college SPOC.
- [ ] Confirm **Theme = Miscellaneous** (the internal-round PDF said "Smart Vehicles"; the official catalogue lists Miscellaneous for SIH26168).
- [ ] Read every slide once; fix any wording you disagree with, but keep it at 6 slides and on the template (SIH rule).
- [ ] Export to PDF too (File → Export) and keep both files. Upload whatever the portal / SPOC asks for (usually PPT or PDF).

## 2. GitHub (do this before anyone opens the link) — needs **Anushtup** (repo admin)

- [ ] Make `AnushtupGhosh5/sih` **public** (Settings → General → Danger zone → Change visibility). It is private today, so the link on the deck shows a 404 to judges.
- [ ] Set the **default branch to `main`** (Settings → Branches). Today it is `calibration-Abhilash`, which only has the alignment stub.
- [ ] After that, open https://github.com/AnushtupGhosh5/sih in a private/incognito window and check the README with figures shows up.

Already done tonight (by Nilesh via Claude):
- README.md with results table, architecture, reproduce steps and figures pushed to `main`.
- `docs/figures/` with the position plots added (artifacts/ was git-ignored, so judges could not see any plot before).
- Release `v0.1-idea-submission` (HMI-only APK) and `v0.2-on-device-engine` (`app-release.apk`: alignment, calibration, dead reckoning and map-matching now run on the phone; `flutter test` passes 18 engine tests).

## 3. Optional but strong (if there is time before the upload)

- [ ] Install `app-release.apk` (v0.2) on an Android phone, drive 30 s with GPS, tap **SIMULATE TUNNEL**, watch the dot keep moving, tap **END TUNNEL** and note the measured drift. Record it; it is the demo.

- [ ] Record a 30–60 s screen capture of `demo/index.html` (the replay) and put the link on slide 6 or in the README.
- [ ] Run `python -m pytest src/alignment src/sensor_fusion` once on a clean clone to be sure tests pass as claimed.
- [ ] Ask Ritu to double-check the numbers on slide 4 against `RESULTS.md` (they were copied from it verbatim).

## 4. What judges look for at screening (from the PS text)

- Preliminary AI models **and position plots inferred from a subset of IO-VNBD** — slide 2 and slide 4 carry these.
- All six expected modules named: alignment/calibration, AI speed & vibration filter, map-matching + NHC, GNSS+INS fusion, GNSS deficit handler, real-time navigation UI — slide 3 covers each.
- Benchmark: < 10 % drift (e.g. < 5 m over 50 m, < 100 m over 1 km), 10 Hz on phone, ~200 Hz on edge engine — slide 4 and slide 5.
- Edge-deployable engine for non-phone IMUs — mentioned on slides 2, 3 and 4.
