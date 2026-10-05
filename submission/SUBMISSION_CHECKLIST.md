# KANABI — what to do, in order (revised 5 Oct 2026, late evening)

Deadline for the idea PPT is tonight. Items 1–3 are the only ones that can
cost us the screening. Everything else is for the finale.

## Tonight (blocking)

1. **Slide 1: Team ID.** Open `KANABI_SIH26168_IdeaPPT.pptx`, replace the red
   `<Team ID from portal>` with the real Team ID. Keep Theme = Miscellaneous.
   Re-export the PDF (File → Export). Upload PPTX or PDF, whichever the portal
   or SPOC asks for. Keep 6 slides; the template is mandatory.
2. **Anushtup (repo admin):** make `AnushtupGhosh5/sih` public and set the
   default branch to `main`. Until then the GitHub link on slide 6 is a 404
   for judges.
3. **Check the link** in an incognito window: README with figures, releases
   `v0.1` and `v0.2` visible.

## This week (first real test of the app)

4. Install `app-release.apk` (v0.2, engine on device) on an Android phone.
   Grant location and sensor permissions. Mount it on the dashboard.
5. Drive 2–3 minutes with GPS. The status strip should show
   `ALIGNED · CAL 0.xx · MAP n.nk`. If it stays `ALIGNING · CAL —` after a few
   accelerations, or `MAP —` after a minute online, note it.
6. While moving at a steady speed, tap **SIMULATE TUNNEL**. The dot must keep
   moving, the trail turns orange, the readout counts metres and seconds.
   After 30–60 s tap **END TUNNEL** and write down the drift line
   (`LAST DR … m · … m OFF (… %)`). Repeat with a turn, and once while stopped
   at a light. Send the numbers and any oddity (dot stuck, wrong speed,
   wrong turn) back to Nilesh so the thresholds can be tuned.
7. If an underpass or flyover underside is nearby, drive through it with the
   app running and screen-record it. That recording is the demo.

## Before screening / finale (ranked by value per day of work)

8. **Demo video** (1 day): dashboard view + screen recording, 60 s, with the
   drift readout at the end. Link on slide 6 and in the README.
9. **CSV logger + own data** (1–2 days): log IMU + GNSS from the app, record
   Kolkata drives (car and scooter), run the desktop pipeline with simulated
   blackouts, add the results next to the IO-VNBD ones.
10. **Edge engine CLI** (1 day): `python -m edge_engine --imu file.csv --rate 200`.
    Replaces the stub, answers "edge deployable, 200 Hz", and handles the
    datasets judges provide at screening.
11. **Bundle the venue map** (half a day): download the OSM road graph of the
    finale city and ship it in the APK so map-matching works offline.
12. **Fusion on the phone** (1–2 days): small Kalman filter for GNSS+INS.
13. **Multi-hypothesis map-matching** (1–2 days) and **two-wheeler lean
    handling** (1 day) if time allows.
14. **Repo hygiene** (1 hour): delete or fill the empty stubs
    (`edge_engine/`, `src/deficit_handler/`, `src/filtering/`,
    `src/map_matching/`), remove committed `__pycache__`, add a LICENSE and a
    GitHub Actions workflow that runs `flutter test` and the Python tests.
15. **Rehearse** the 6 slides in under 5 minutes and the Q&A list in
    `REVIEW_2026-10-05.md` §4.8. Everyone must be able to explain why heading
    is the hard part and what map-matching does.

## Already done (tonight)

- Official-template deck (6 slides) with IO-VNBD position plots and numbers.
- README with results, architecture and figures on `main`; `docs/figures/`.
- Engine ported to the phone (pure Dart), 20 engine tests passing, review
  fixes applied (see `REVIEW_2026-10-05.md` §2).
- Releases `v0.1-idea-submission` (HMI only) and `v0.2-on-device-engine`.
