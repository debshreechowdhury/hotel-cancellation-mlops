"""Streamlit front end for the Hotel Cancellation Predictor.

Run locally:
    streamlit run frontend/app.py
Set the API URL with env var API_URL, or in .streamlit/secrets.toml:
    API_URL = "https://<your-app>.azurewebsites.net"
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Will this guest cancel?", page_icon="🏨", layout="wide")

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
SEGMENTS = ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation"]
CHANNELS = ["TA/TO", "Direct", "Corporate", "GDS"]
ROOMS = ["A", "B", "C", "D", "E", "F", "G", "H", "L"]
RISK_COLORS = {"low": "#2e7d32", "medium": "#ef6c00", "high": "#c62828"}

# ---------------------------------------------------------------------------
# Example bookings users can load with one click.
# Keys must match the widget keys used in the form below.
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    "Typical online booking": {
        "hotel": "City Hotel", "lead_time": 90, "arrival_date_month": "August",
        "stays_in_weekend_nights": 1, "stays_in_week_nights": 2, "adults": 2,
        "children": 0, "babies": 0, "meal": "BB", "market_segment": "Online TA",
        "distribution_channel": "TA/TO", "reserved_room_type": "A",
        "deposit_type": "No Deposit", "customer_type": "Transient", "adr": 110.0,
        "is_repeated_guest": False, "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0, "booking_changes": 0,
        "days_in_waiting_list": 0, "required_car_parking_spaces": 0,
        "total_of_special_requests": 1,
    },
    "Risky group booking": {
        "hotel": "City Hotel", "lead_time": 300, "arrival_date_month": "October",
        "stays_in_weekend_nights": 0, "stays_in_week_nights": 2, "adults": 2,
        "children": 0, "babies": 0, "meal": "BB", "market_segment": "Groups",
        "distribution_channel": "TA/TO", "reserved_room_type": "A",
        "deposit_type": "Non Refund", "customer_type": "Transient-Party", "adr": 85.0,
        "is_repeated_guest": False, "previous_cancellations": 1,
        "previous_bookings_not_canceled": 0, "booking_changes": 0,
        "days_in_waiting_list": 0, "required_car_parking_spaces": 0,
        "total_of_special_requests": 0,
    },
    "Loyal business traveller": {
        "hotel": "City Hotel", "lead_time": 7, "arrival_date_month": "March",
        "stays_in_weekend_nights": 0, "stays_in_week_nights": 2, "adults": 1,
        "children": 0, "babies": 0, "meal": "BB", "market_segment": "Corporate",
        "distribution_channel": "Corporate", "reserved_room_type": "A",
        "deposit_type": "No Deposit", "customer_type": "Transient", "adr": 95.0,
        "is_repeated_guest": True, "previous_cancellations": 0,
        "previous_bookings_not_canceled": 5, "booking_changes": 0,
        "days_in_waiting_list": 0, "required_car_parking_spaces": 1,
        "total_of_special_requests": 1,
    },
}


def get_api_url() -> str:
    try:
        url = st.secrets.get("API_URL")  # Streamlit Cloud secrets
    except Exception:
        url = None
    return (url or os.environ.get("API_URL") or "http://localhost:8080").rstrip("/")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_metadata(api_url: str) -> dict | None:
    try:
        r = requests.get(f"{api_url}/metadata", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def call_api(api_url: str, path: str, payload: dict) -> dict:
    r = requests.post(f"{api_url}{path}", json=payload, timeout=60)
    if r.status_code == 422:
        raise ValueError(f"Invalid input: {r.json().get('detail')}")
    r.raise_for_status()
    return r.json()


def apply_preset(name: str) -> None:
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


# ------------------------------- Sidebar -----------------------------------
api_url = get_api_url()
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API URL", api_url).rstrip("/")
    st.caption("The first request can take ~10–20 s while the serverless API wakes up.")
    st.divider()
    st.subheader("Load an example")
    preset = st.selectbox("Preset booking", list(PRESETS))
    st.button("Load preset", on_click=apply_preset, args=(preset,))
    st.divider()
    meta = fetch_metadata(api_url)
    if meta:
        st.subheader("Model")
        st.write(f"**{meta.get('model_type')}**  \n`{meta.get('model_version')}`")
        m = meta.get("metrics") or {}
        c1, c2 = st.columns(2)
        c1.metric("Test ROC-AUC", f"{m.get('test/roc_auc', 0):.3f}")
        c2.metric("Test F1", f"{m.get('test/f1', 0):.3f}")
        if meta.get("wandb_run_url"):
            st.markdown(f"[W&B run ↗]({meta['wandb_run_url']})")
    else:
        st.warning("Can't reach the API — check the URL.")

# ------------------------------- Main form ---------------------------------
st.title("🏨 Will this guest cancel?")
st.write("Enter a booking's details to get the predicted probability that it will be "
         "cancelled. The prediction comes from a live model API deployed on Azure Functions.")

if "lead_time" not in st.session_state:  # first load -> default preset
    apply_preset(next(iter(PRESETS)))

with st.form("booking"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Stay**")
        st.selectbox("Hotel", ["City Hotel", "Resort Hotel"], key="hotel")
        st.selectbox("Arrival month", MONTHS, key="arrival_date_month")
        st.number_input("Lead time (days before arrival)", 0, 700, key="lead_time")
        st.number_input("Weekend nights", 0, 20, key="stays_in_weekend_nights")
        st.number_input("Week nights", 0, 50, key="stays_in_week_nights")
        st.number_input("Average daily rate (€)", 0.0, 1000.0, step=5.0, key="adr")
    with c2:
        st.markdown("**Guests**")
        st.number_input("Adults", 0, 10, key="adults")
        st.number_input("Children", 0, 10, key="children")
        st.number_input("Babies", 0, 10, key="babies")
        st.selectbox("Customer type", ["Transient", "Transient-Party", "Contract", "Group"],
                     key="customer_type")
        st.checkbox("Repeat guest", key="is_repeated_guest")
        st.number_input("Previous cancellations", 0, 30, key="previous_cancellations")
        st.number_input("Previous stays (not cancelled)", 0, 80, key="previous_bookings_not_canceled")
    with c3:
        st.markdown("**Booking**")
        st.selectbox("Deposit", ["No Deposit", "Non Refund", "Refundable"], key="deposit_type")
        st.selectbox("Market segment", SEGMENTS, key="market_segment")
        st.selectbox("Distribution channel", CHANNELS, key="distribution_channel")
        st.selectbox("Room type", ROOMS, key="reserved_room_type")
        st.selectbox("Meal plan", ["BB", "HB", "FB", "SC"], key="meal",
                     help="BB=bed & breakfast, HB=half board, FB=full board, SC=no meal")
        st.number_input("Special requests", 0, 5, key="total_of_special_requests")
        st.number_input("Parking spaces", 0, 3, key="required_car_parking_spaces")
        st.number_input("Booking changes", 0, 20, key="booking_changes")
        st.number_input("Days on waiting list", 0, 400, key="days_in_waiting_list")
    submitted = st.form_submit_button("Predict cancellation risk", type="primary")

FIELDS = list(PRESETS[next(iter(PRESETS))].keys())


def current_booking() -> dict:
    b = {k: st.session_state[k] for k in FIELDS}
    b["is_repeated_guest"] = int(bool(b["is_repeated_guest"]))
    b["adr"] = float(b["adr"])
    return b


if submitted:
    booking = current_booking()
    try:
        with st.spinner("Asking the model..."):
            pred = call_api(api_url, "/predict", booking)
    except (requests.RequestException, ValueError) as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    p = pred["cancel_probability"]
    color = RISK_COLORS[pred["risk_level"]]
    st.divider()
    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f"<div style='padding:1.2rem;border-radius:12px;background:{color}15;"
            f"border:2px solid {color};text-align:center'>"
            f"<div style='font-size:0.95rem;color:{color}'>Cancellation probability</div>"
            f"<div style='font-size:3rem;font-weight:700;color:{color}'>{p:.0%}</div>"
            f"<div style='font-size:1.1rem;color:{color}'>{pred['risk_level'].upper()} RISK</div>"
            f"</div>", unsafe_allow_html=True)
        st.progress(min(max(p, 0.0), 1.0))
        verdict = "likely to **cancel**" if pred["will_cancel"] else "likely to **show up**"
        st.write(f"At a {pred['threshold']:.0%} threshold this booking is {verdict}.")

    # What-if: how does risk change with lead time? (one batch request)
    with right:
        st.markdown("**What if the booking were made earlier or later?**")
        leads = [0, 7, 14, 30, 60, 90, 120, 180, 240, 300, 365]
        batch = [{**booking, "lead_time": lt} for lt in leads]
        try:
            res = call_api(api_url, "/predict/batch", {"bookings": batch})
            curve = pd.DataFrame({"Lead time (days)": leads,
                                  "Cancellation probability":
                                      [r["cancel_probability"] for r in res["predictions"]]})
            st.line_chart(curve, x="Lead time (days)", y="Cancellation probability", height=260)
        except (requests.RequestException, ValueError) as e:
            st.info(f"What-if chart unavailable: {e}")

    with st.expander("Raw API request / response"):
        st.json({"request": booking, "response": pred})

st.caption("Model trained on the Hotel Booking Demand dataset (Antonio, Almeida & Nunes, 2019). "
           "Predictions are probabilistic estimates for decision support, not decisions about guests.")
