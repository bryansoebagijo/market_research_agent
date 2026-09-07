"""Thesis monitoring: the views you've saved, and whether they still hold."""

import streamlit as st

from agent import market_data as md, theses
from ui_common import money

st.set_page_config(page_title="Theses · Markets Dashboard", page_icon="📌", layout="wide")

STATUS_STYLE = {
    "broken": ("🔴", "Thesis broken", "A condition you said would change your mind has been met."),
    "intact": ("🟢", "Holding", "Every measurable condition is still within bounds."),
    "unverified": ("⚪", "Not verified",
                   "Nothing could be checked automatically — this one needs your judgement."),
}
CHECK_ICONS = {"holding": "✅", "triggered": "🔴", "unknown": "❔"}


@st.cache_data(ttl=600, show_spinner=False)
def cached_metrics(symbol: str):
    return md.current_metrics(symbol)


st.title("📌 Theses")
st.caption(
    "A saved view plus the conditions that would break it. Rather than re-analysing "
    "everything, decide what would prove you wrong and watch only for that — the "
    "measurable conditions are checked automatically against current data."
)

saved = theses.list_theses()
if not saved:
    st.info(
        "No theses saved yet. Run **Analyze** on a coin or stock, then press "
        "**Save as thesis** on the result — the analyst's invalidation conditions become "
        "the things monitored here."
    )
    st.stop()

# ---------------- summary ----------------
results = {}
with st.spinner("Checking conditions against current data…"):
    for t in saved:
        results[t["id"]] = theses.evaluate(t, cached_metrics(t["symbol"]))

broken = [t for t in saved if results[t["id"]]["status"] == "broken"]
stale = [t for t in saved if results[t["id"]]["stale"]]

c1, c2, c3 = st.columns(3)
c1.metric("Theses tracked", len(saved))
c2.metric("Broken", len(broken), delta="needs review" if broken else None,
          delta_color="inverse" if broken else "off")
c3.metric("Stale", len(stale), help=f"Formed more than {theses.STALE_AFTER_DAYS} days ago")

if broken:
    st.error(
        "**" + ", ".join(t["symbol"] for t in broken) + "** — a condition you set has "
        "been met. That's the moment to revisit the view, not the price."
    )

st.divider()

# ---------------- one card per thesis ----------------
for t in saved:
    r = results[t["id"]]
    icon, title, blurb = STATUS_STYLE[r["status"]]
    age = r["age_days"]
    age_text = "today" if age == 0 else f"{age} day{'s' if age != 1 else ''} ago"

    with st.container(border=True):
        head, meta = st.columns([3, 2])
        head.markdown(f"### {icon} {t['symbol']} — {t['stance']}")
        head.caption(blurb)
        meta.markdown(
            f"**Conviction** {t['conviction'] or '—'}  \n"
            f"**Horizon** {t['horizon'] or '—'}  \n"
            f"**Formed** {age_text}" + ("  ⏳ *stale*" if r["stale"] else "")
        )

        if r["stale"]:
            st.warning(
                f"Formed {age_text}, on data that has since moved. Worth re-running the "
                "analysis rather than trusting this as current."
            )

        # --- the monitored conditions ---
        if r["checks"]:
            st.markdown("**Conditions being watched**")
            for c in r["checks"]:
                cols = st.columns([5, 2, 2])
                cols[0].markdown(f"{CHECK_ICONS[c['status']]} {c['description']}")
                cols[1].markdown(f"`{c['label']}`  \nnow **{c['current_text']}**")
                cols[2].markdown(f"trips if  \n`{c['operator']} {c['threshold_text']}`")

        if r["manual"]:
            st.markdown("**Needs your judgement** — not reducible to a number")
            for m in r["manual"]:
                st.markdown(f"❔ {m}")

        with st.expander("The original view"):
            if t.get("quality"):
                st.markdown(f"**Quality** — {t['quality']}")
            if t.get("valuation"):
                st.markdown(f"**Valuation** — {t['valuation']}")
            if t.get("reasoning"):
                st.markdown("**Why**")
                for x in t["reasoning"]:
                    st.markdown(f"- {x}")
            if t.get("key_risks"):
                st.markdown("**Key risks**")
                for x in t["key_risks"]:
                    st.markdown(f"- {x}")

        if st.button("Delete thesis", key=f"del_{t['id']}"):
            theses.remove_thesis(t["id"])
            st.rerun()

st.caption(
    "Conditions are evaluated deterministically from current data — same numbers, same "
    "answer. Informational only, not advice."
)
