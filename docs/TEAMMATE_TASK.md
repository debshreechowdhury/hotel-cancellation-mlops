# Teammate PR: `feature/presets-and-ethics`

A small, self-contained PR (about 1–2 hours). It only touches three files, so it
can't break the pipeline or the API.

## Setup
```bash
git clone <repo-url> && cd <repo>
git checkout feature/presets-and-ethics     # branch already exists
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Task 1 — Add example presets to the Streamlit app (`frontend/app.py`)

Find the `PRESETS` dict near the top (marked `TODO(teammate PR)`). Add **at least two**
more presets by copying the existing entry and changing values. Keep **all the same keys**.

Suggested presets:
- **"Risky last-minute group"**: `market_segment: "Groups"`, `deposit_type: "Non Refund"`, `lead_time: 300`, `customer_type: "Transient-Party"`, `total_of_special_requests: 0`
- **"Loyal business traveller"**: `market_segment: "Corporate"`, `distribution_channel: "Corporate"`, `is_repeated_guest: True`, `previous_bookings_not_canceled: 5`, `lead_time: 7`, `required_car_parking_spaces: 1`

Allowed values: see the selectbox lists in the same file (`SEGMENTS`, `CHANNELS`, `ROOMS`, `MONTHS`).

Test it:
```bash
API_URL=https://<our-api>.azurewebsites.net streamlit run frontend/app.py
```
Pick each preset in the sidebar → **Load preset** → **Predict** and check that it works.

## Task 2 — Fill in model results and write "Ethics & limitations" (`README.md`)

1. In section 3, fill the results table from our W&B dashboard (ask for the link).
2. Replace the TODO in **section 7** with ~200–300 words covering:
   - **Proxy / fairness risk:** `country` predicts cancellation well but stands in for nationality. We left it out by default (`features.include_country: false`). Mention how much AUC it adds if we ran that experiment.
   - **How predictions could be misused:** e.g. charging or refusing guests from "high-risk" groups, versus legitimate uses like overbooking buffers and reminder emails.
   - **Data limitations:** two hotels in Portugal, 2015–2017, pre-COVID. Won't generalise to other markets or to today's booking behaviour.
   - **Model limitations:** threshold 0.5 is arbitrary; probabilities aren't calibrated; synthetic test data is only for CI.
   - **Human in the loop:** predictions support staff decisions; they don't make them.
3. In section 9, add both names.

## Task 3 — Open the PR
```bash
git add frontend/app.py README.md
git commit -m "Add booking presets and ethics & limitations section"
git push -u origin feature/presets-and-ethics
```
On GitHub: **Compare & pull request** → base `main` → describe what you changed →
request your teammate as reviewer. Wait for the green CI check, then merge.
