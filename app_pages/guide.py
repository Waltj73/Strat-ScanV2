# app_pages/guide.py

import streamlit as st

def guide_main():
    st.title("📘 STRAT App Guide (V1)")
    st.caption("How to use this STRAT-only app without overthinking.")

    st.markdown("""
## What this app is (and is NOT)

✅ **It IS**
- A STRAT-only market regime + sector rotation + trigger finder  
- Built to answer: **“What’s aligned, and what’s actionable today?”**

❌ **It is NOT**
- RSI / indicators / mixed systems  
- A prediction machine  
- A “force trades every day” tool

---

## Page-by-page

### 1) Scanner
**Purpose:** Find the best ideas based on market bias and inside-bar triggers.

**What to look for:**
- **Bias** (LONG / SHORT / MIXED)
- **Rotation IN / OUT** (which groups are strongest/weakest in STRAT terms)
- In the drilldown table: **TriggerStatus**
  - `W: READY` = top-tier
  - `D: READY` = still valid
  - no READY = wait

**Rule:** No trigger = no trade.

---

### 2) Ticker Analyzer
**Purpose:** One ticker, fully explained — STRAT snapshot + actionable trigger.

You get:
- Candle Types (D/W/M)
- Flags (Inside, Bull/Bear alignment, 2-1-2)
- **Entry/Stop** if an Inside Bar exists
- ATR%, Targets (20d/63d), and RR to T2 (simple planning)

---

### 3) Cheat Sheet
**Purpose:** Definitions so you trust what you’re seeing.

---

## The Golden Rules
1) **Weekly triggers > Daily triggers**
2) If bias is LONG, prioritize long setups. If SHORT, prioritize short setups.
3) Don’t “manufacture” trades — wait for the bar that gives you the trigger.
4) Inside bar gives you the cleanest execution: **entry + stop from the same bar.**

---

## Common mistakes
- Trading “good looking” tickers with **no trigger**
- Ignoring bias and fighting the tape
- Over-scanning instead of waiting for the right bar

---

## Your simplest daily plan
- Bias → Rotation IN → Find READY → Place order → Done
""")
