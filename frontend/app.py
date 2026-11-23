"""Streamlit front end for the Hotel Cancellation Predictor.

Run locally:
    streamlit run frontend/app.py
Set the API URL with env var API_URL, or in .streamlit/secrets.toml:
    API_URL = "https://<your-app>.azurewebsites.net"
"""
from __future__ import annotations

import html
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Will this guest cancel?", page_icon="🛎️", layout="wide")

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
SEGMENTS = ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation"]
CHANNELS = ["TA/TO", "Direct", "Corporate", "GDS"]
ROOMS = ["A", "B", "C", "D", "E", "F", "G", "H", "L"]
MEALS = {"BB": "Bed & breakfast", "HB": "Half board", "FB": "Full board", "SC": "Room only"}

# Palette: hotel lobby green, brass, cool paper, ink, and three stamp inks
LOBBY, LOBBY_DEEP, BRASS = "#1F5446", "#12352E", "#B08D57"
PAPER, INK, MUTED = "#F3F6F4", "#1B2421", "#5B6B65"
RISK = {
    "low": {"ink": "#2F7D5B", "stamp": "Likely to arrive",
            "advice": "No action needed. Treat it as a confirmed stay."},
    "medium": {"ink": "#B7791F", "stamp": "Worth a reminder",
               "advice": "Send a pre-arrival reminder or ask the guest to confirm."},
    "high": {"ink": "#B3342B", "stamp": "Likely to cancel",
             "advice": "Plan for this room to come back. Consider an overbooking buffer or a deposit."},
}
FEATURE_NAMES = {
    "lead_time": "How far ahead it was booked",
    "required_car_parking_spaces": "Parking requested",
    "total_of_special_requests": "Special requests",
    "market_segment": "Where the booking came from",
    "adr": "Nightly rate",
    "deposit_type": "Deposit type",
    "previous_cancellations": "Guest's past cancellations",
    "customer_type": "Type of booking",
    "arrival_date_month": "Arrival month",
    "booking_changes": "Changes to the booking",
    "distribution_channel": "Sales channel",
    "hotel": "Hotel",
    "stays_in_week_nights": "Week nights",
    "stays_in_weekend_nights": "Weekend nights",
    "is_repeated_guest": "Returning guest",
    "previous_bookings_not_canceled": "Guest's past stays",
    "reserved_room_type": "Room type",
    "meal": "Meal plan",
    "adults": "Adults",
    "children": "Children",
    "days_in_waiting_list": "Days on the waiting list",
}

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
FIELDS = list(next(iter(PRESETS.values())).keys())

# ------------------------------------------------------------------- styles
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Marcellus&family=Karla:wght@400;500;700&display=swap');

.stApp {{ background: {PAPER}; color: {INK}; font-family: 'Karla', system-ui, sans-serif; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1180px; }}
h1, h2, h3, .serif {{ font-family: 'Marcellus', Georgia, serif !important; font-weight: 400 !important;
                       letter-spacing: 0.01em; }}

/* Lobby header */
.lobby {{ background: {LOBBY_DEEP}; color: {PAPER}; border-radius: 4px; padding: 2.4rem 2.6rem 2rem;
         position: relative; overflow: hidden; margin-bottom: 1.6rem; }}
.lobby::after {{ content: ""; position: absolute; inset: 10px; border: 1px solid {BRASS}55;
                border-radius: 2px; pointer-events: none; }}
.lobby h1 {{ color: {PAPER}; font-size: clamp(2.2rem, 4.6vw, 3.6rem); line-height: 1.05; margin: 0 0 .6rem; padding: 0; }}
.lobby p {{ color: #C9D6D1; font-size: 1.08rem; max-width: 34rem; margin: 0 0 1.1rem; }}
.lobby .status {{ color: {BRASS}; font-size: .95rem; margin: 0; }}
.lobby .status b {{ color: {PAPER}; font-weight: 500; }}
.lobby .bell {{ position: absolute; right: 2.4rem; bottom: -0.6rem; font-size: 7.5rem; opacity: .13;
               filter: grayscale(1); user-select: none; }}

/* Form */
[data-testid="stForm"] {{ background: #FFFFFF; border: 1px solid #D5E0DB; border-radius: 4px; padding: 1.1rem 1.3rem 1.3rem; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'Marcellus', Georgia, serif; font-size: 1.05rem; }}
.stTabs [aria-selected="true"] {{ color: {LOBBY} !important; }}
[data-testid="stFormSubmitButton"] button {{ width: 100%; background: {LOBBY}; border: 1px solid {LOBBY}; color: {PAPER};
    font-weight: 700; padding: .7rem 1rem; border-radius: 3px; }}
[data-testid="stFormSubmitButton"] button:hover {{ background: {LOBBY_DEEP}; border-color: {BRASS}; color: #FFFFFF; }}
.hint {{ color: {MUTED}; font-size: .92rem; margin: .2rem 0 .6rem; }}

/* Reservation slip */
.slip {{ background: #FFFFFF; border: 1px solid #D5E0DB; border-radius: 4px; padding: 1.4rem 1.5rem 1.2rem;
        position: relative; overflow: hidden; }}
.slip::before {{ content: ""; position: absolute; left: 0; right: 0; top: 0; height: 6px;
                background: repeating-linear-gradient(90deg, {BRASS} 0 14px, transparent 14px 22px); opacity: .55; }}
.slip h3 {{ margin: .2rem 0 .1rem; font-size: 1.45rem; color: {INK}; }}
.slip .sub {{ color: {MUTED}; margin: 0 0 1rem; font-size: .95rem; }}
.slip dl {{ display: grid; grid-template-columns: auto 1fr; gap: .25rem 1rem; margin: 0 0 1.1rem;
           font-size: .95rem; }}
.slip dt {{ color: {MUTED}; }}
.slip dd {{ margin: 0; color: {INK}; font-weight: 500; }}
.slip .prob {{ font-family: 'Marcellus', Georgia, serif; font-size: 3.4rem; line-height: 1; margin: .2rem 0 .1rem; }}
.slip .prob-label {{ color: {MUTED}; font-size: .95rem; margin-bottom: .7rem; }}
.meter {{ position: relative; height: 12px; background: #E4ECE8; border-radius: 6px; margin: .4rem 0 1.4rem; }}
.meter .fill {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 6px; }}
.meter .line {{ position: absolute; top: -5px; bottom: -5px; width: 2px; background: {INK}; }}
.meter .line span {{ position: absolute; top: 18px; left: -2.6rem; width: 5.2rem; text-align: center;
                    font-size: .78rem; color: {MUTED}; }}
.advice {{ border-left: 3px solid; padding: .1rem 0 .1rem .8rem; margin: .4rem 0 0; font-size: 1rem; }}
.stamp {{ position: absolute; right: 1.1rem; top: 2.4rem; transform: rotate(-9deg);
         border: 3px double; border-radius: 6px; padding: .35rem .8rem; font-family: 'Marcellus', Georgia, serif;
         font-size: 1.15rem; letter-spacing: .04em; background: #FFFFFFcc; animation: land .45s cubic-bezier(.2,1.4,.4,1) both; }}
.stamp.empty {{ color: #A7B5AF; border-color: #C3CFCA; border-style: dashed; animation: none; }}
@keyframes land {{ from {{ transform: rotate(-9deg) scale(1.6); opacity: 0; }} to {{ transform: rotate(-9deg) scale(1); opacity: 1; }} }}
@media (prefers-reduced-motion: reduce) {{ .stamp {{ animation: none; }} }}
@media (max-width: 640px) {{ .stamp {{ position: static; display: inline-block; margin-bottom: .8rem; }}
                            .lobby {{ padding: 1.8rem 1.4rem 1.6rem; }} .lobby .bell {{ display: none; }} }}

/* Drivers */
.drivers {{ margin-top: 1.2rem; }}
.drivers h4 {{ font-family: 'Marcellus', Georgia, serif; font-weight: 400; font-size: 1.1rem; margin: 0 0 .2rem; }}
.drivers .note {{ color: {MUTED}; font-size: .85rem; margin: 0 0 .6rem; }}
.drv {{ display: grid; grid-template-columns: 11.5rem 1fr; align-items: center; gap: .7rem; margin: .3rem 0;
       font-size: .92rem; }}
.drv .bar {{ height: 8px; background: {LOBBY}; border-radius: 4px; opacity: .8; }}
@media (max-width: 640px) {{ .drv {{ grid-template-columns: 8.5rem 1fr; }} }}
[data-testid="stVegaLiteChart"] {{ border: 1px solid #D5E0DB; border-radius: 4px; overflow: hidden; margin-top: .9rem; }}
.foot {{ color: {MUTED}; font-size: .85rem; margin-top: 2rem; max-width: 46rem; }}
</style>
""", unsafe_allow_html=True)


def show_html(markup: str) -> None:
    """Render raw HTML. Lines are de-indented first: markdown treats 4+ leading
    spaces as a code block, which would print the HTML instead of rendering it."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines() if line.strip()),
                unsafe_allow_html=True)


# ------------------------------------------------------------------- API helpers
def default_api_url() -> str:
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
        raise ValueError(f"The API rejected some fields: {r.json().get('detail')}")
    r.raise_for_status()
    return r.json()


def apply_preset() -> None:
    name = st.session_state.get("preset_choice")
    if name in PRESETS:
        for k, v in PRESETS[name].items():
            st.session_state[k] = v
        st.session_state.pop("result", None)


def current_booking() -> dict:
    b = {k: st.session_state[k] for k in FIELDS}
    b["is_repeated_guest"] = int(bool(b["is_repeated_guest"]))
    b["adr"] = float(b["adr"])
    return b


if "lead_time" not in st.session_state:  # first visit -> first preset
    for k, v in next(iter(PRESETS.values())).items():
        st.session_state[k] = v
if "api_url" not in st.session_state:
    st.session_state["api_url"] = default_api_url()
api_url = st.session_state["api_url"].rstrip("/")
meta = fetch_metadata(api_url)

# ------------------------------------------------------------------- header
if meta:
    m = meta.get("metrics") or {}
    model_name = (meta.get("model_type") or "model").replace("_", " ")
    status = (f"Live model: <b>{html.escape(model_name)}</b>, scoring "
              f"<b>{m.get('test/roc_auc', 0):.3f}</b> ROC-AUC on bookings it never saw. API connected.")
else:
    status = "The prediction API isn't answering yet. It may be waking up; reload in a few seconds."
show_html(f"""
<div class="lobby">
  <h1>Will this guest cancel?</h1>
  <p>Check a reservation's cancellation risk before you count on the room. Fill in the booking, or start from an example.</p>
  <p class="status">{status}</p>
  <div class="bell" aria-hidden="true">🛎️</div>
</div>
""")

left, right = st.columns([7, 5], gap="large")

# ------------------------------------------------------------------- form
with left:
    st.segmented_control("Start from an example", list(PRESETS), key="preset_choice",
                         on_change=apply_preset)
    with st.form("booking", border=False):
        stay, guests, history = st.tabs(["The stay", "The guests", "Booking details"])
        with stay:
            c1, c2 = st.columns(2)
            c1.selectbox("Hotel", ["City Hotel", "Resort Hotel"], key="hotel")
            c2.selectbox("Arrival month", MONTHS, key="arrival_date_month")
            st.slider("Booked how many days ahead?", 0, 700, key="lead_time")
            c1, c2, c3 = st.columns(3)
            c1.number_input("Weekend nights", 0, 20, key="stays_in_weekend_nights")
            c2.number_input("Week nights", 0, 50, key="stays_in_week_nights")
            c3.number_input("Nightly rate (€)", 0.0, 1000.0, step=5.0, key="adr")
            c1, c2 = st.columns(2)
            c1.selectbox("Room type", ROOMS, key="reserved_room_type")
            c2.selectbox("Meal plan", list(MEALS), key="meal", format_func=MEALS.get)
        with guests:
            c1, c2, c3 = st.columns(3)
            c1.number_input("Adults", 0, 10, key="adults")
            c2.number_input("Children", 0, 10, key="children")
            c3.number_input("Babies", 0, 10, key="babies")
            c1, c2 = st.columns(2)
            c1.selectbox("Type of booking", ["Transient", "Transient-Party", "Contract", "Group"],
                         key="customer_type")
            c2.checkbox("Returning guest", key="is_repeated_guest")
            c1, c2 = st.columns(2)
            c1.number_input("Past cancellations", 0, 30, key="previous_cancellations")
            c2.number_input("Past stays completed", 0, 80, key="previous_bookings_not_canceled")
        with history:
            c1, c2 = st.columns(2)
            c1.selectbox("Deposit", ["No Deposit", "Non Refund", "Refundable"], key="deposit_type")
            c2.selectbox("Booked through", SEGMENTS, key="market_segment")
            c1, c2 = st.columns(2)
            c1.selectbox("Sales channel", CHANNELS, key="distribution_channel")
            c2.number_input("Days on the waiting list", 0, 400, key="days_in_waiting_list")
            c1, c2, c3 = st.columns(3)
            c1.number_input("Special requests", 0, 5, key="total_of_special_requests")
            c2.number_input("Parking spaces", 0, 3, key="required_car_parking_spaces")
            c3.number_input("Changes made", 0, 20, key="booking_changes")
        st.markdown('<p class="hint">All three tabs are sent together. Untouched fields keep their example values.</p>',
                    unsafe_allow_html=True)
        submitted = st.form_submit_button("Check cancellation risk", type="primary")

if submitted:
    booking = current_booking()
    try:
        with st.spinner("Asking the model. The first check after a quiet spell can take about 15 seconds."):
            pred = call_api(api_url, "/predict", booking)
            leads = [0, 7, 14, 30, 60, 90, 120, 180, 240, 300, 365, 500]
            batch = call_api(api_url, "/predict/batch",
                             {"bookings": [{**booking, "lead_time": lt} for lt in leads]})
        st.session_state["result"] = {
            "booking": booking, "pred": pred,
            "curve": [{"lead": lt, "p": r["cancel_probability"]} for lt, r in zip(leads, batch["predictions"])],
        }
    except (requests.RequestException, ValueError) as e:
        st.session_state["result"] = {"error": str(e)}


# ------------------------------------------------------------------- slip
def pct(p: float) -> str:
    """Never claim certainty: the model's extremes are shown as >99% / <1%."""
    return ">99%" if p >= 0.995 else "<1%" if p < 0.005 else f"{p:.0%}"


def facts_html(b: dict) -> str:
    nights = b["stays_in_weekend_nights"] + b["stays_in_week_nights"]
    guests = b["adults"] + b["children"] + b["babies"]
    rows = [
        ("Arriving", f"{b['arrival_date_month']}, booked {b['lead_time']} days ahead"),
        ("Stay", f"{nights} night{'s' if nights != 1 else ''}, {guests} guest{'s' if guests != 1 else ''}"),
        ("Rate", f"€{b['adr']:.0f} a night, {MEALS.get(b['meal'], b['meal']).lower()}"),
        ("Deposit", b["deposit_type"].lower().capitalize()),
        ("Booked through", b["market_segment"]),
    ]
    return "<dl>" + "".join(f"<dt>{k}</dt><dd>{html.escape(str(v))}</dd>" for k, v in rows) + "</dl>"


with right:
    result = st.session_state.get("result")
    if not result:
        b = current_booking()
        show_html(f"""
        <div class="slip">
          <div class="stamp empty">Not checked yet</div>
          <h3>{html.escape(b['hotel'])}</h3>
          <p class="sub">Reservation slip</p>
          {facts_html(b)}
          <p class="prob-label">Press <b>Check cancellation risk</b> to stamp this reservation.</p>
        </div>""")
    elif "error" in result:
        show_html(f"""
        <div class="slip">
          <h3>Couldn't reach the model</h3>
          <p class="sub">{html.escape(result['error'])}</p>
          <p class="prob-label">Check the API address under <b>Connection</b> below, then try again.</p>
        </div>""")
    else:
        b, pred = result["booking"], result["pred"]
        p, level, thr = pred["cancel_probability"], pred["risk_level"], pred["threshold"]
        r = RISK[level]
        show_html(f"""
        <div class="slip">
          <div class="stamp" style="color:{r['ink']};border-color:{r['ink']}">{r['stamp']}</div>
          <h3>{html.escape(b['hotel'])}</h3>
          <p class="sub">Reservation slip</p>
          {facts_html(b)}
          <div class="prob" style="color:{r['ink']}">{html.escape(pct(p))}</div>
          <div class="prob-label">chance this booking is cancelled</div>
          <div class="meter" role="img" aria-label="{html.escape(pct(p))} cancellation probability; decision line at {thr:.0%}">
            <div class="fill" style="width:{p*100:.1f}%;background:{r['ink']}"></div>
            <div class="line" style="left:{thr*100:.1f}%"><span>decision line</span></div>
          </div>
          <p class="advice" style="border-color:{r['ink']}">{r['advice']}</p>
        </div>""")

        # What-if chart
        curve = pd.DataFrame(result["curve"])
        now = pd.DataFrame([{"lead": b["lead_time"], "p": p}])
        base = alt.Chart(curve).encode(
            x=alt.X("lead:Q", title="Days booked ahead"),
            y=alt.Y("p:Q", title="Chance of cancelling", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%")),
        )
        chart = (
            base.mark_area(color=LOBBY, opacity=0.08)
            + base.mark_line(color=LOBBY, strokeWidth=2.5)
            + alt.Chart(pd.DataFrame([{"t": thr}])).mark_rule(color=INK, strokeDash=[4, 4]).encode(y="t:Q")
            + alt.Chart(now).mark_point(size=140, filled=True, color=r["ink"], opacity=1).encode(
                x="lead:Q", y="p:Q", tooltip=[alt.Tooltip("lead:Q", title="This booking, days ahead"),
                                              alt.Tooltip("p:Q", title="Chance", format=".0%")])
        ).properties(width="container", height=220, title=alt.TitleParams(
            "If the same booking were made earlier or later", font="Karla", fontSize=15,
            fontWeight=600, color=INK, anchor="start")
        ).configure_axis(labelFont="Karla", titleFont="Karla", labelColor=MUTED, titleColor=MUTED,
                         gridColor="#E4ECE8", domainColor="#C3CFCA"
        ).configure_view(strokeWidth=0).configure(background="#FFFFFF", padding=12)
        st.altair_chart(chart, theme=None)

        with st.expander("See the API call behind this result"):
            st.caption(f"POST {api_url}/predict")
            st.json({"request": b, "response": pred})

    # Global drivers (from the model's metadata, not specific to this booking)
    if meta and meta.get("top_features"):
        top = meta["top_features"][:5]
        mx = max(f["importance"] for f in top) or 1
        rows = "".join(
            f'<div class="drv"><span>{html.escape(FEATURE_NAMES.get(f["feature"], f["feature"]))}</span>'
            f'<div class="bar" style="width:{max(f["importance"] / mx, 0.04) * 100:.0f}%"></div></div>'
            for f in top)
        show_html(f"""
        <div class="drivers">
          <h4>What the model looks at most</h4>
          <p class="note">Across all bookings, measured by how much accuracy drops when each detail is scrambled.</p>
          {rows}
        </div>""")

# ------------------------------------------------------------------- footer
with st.expander("Connection"):
    st.text_input("Prediction API address", key="api_url",
                  help="Set by the app's API_URL secret. Change it to point at a local API.")
    if meta and meta.get("wandb_run_url"):
        st.markdown(f"Training run for this model: [Weights & Biases]({meta['wandb_run_url']})")

st.markdown('<p class="foot">Trained on the Hotel Booking Demand dataset (Antonio, Almeida &amp; Nunes, 2019): '
            'two Portuguese hotels, 2015–2017. Use the result to plan, not to judge guests; '
            'it is a probability, not a decision.</p>', unsafe_allow_html=True)
