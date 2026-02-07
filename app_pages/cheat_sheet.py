# app_pages/cheat_sheet.py

import streamlit as st

def cheat_sheet_main():
    st.title("🧾 STRAT Cheat Sheet (This App)")
    st.caption("Clear definitions + exactly what your scanner/analyzer is labeling.")

    st.markdown("""
## 1) Candle Types (Core STRAT)

### **1 = Inside Bar**
- High ≤ prior High **AND** Low ≥ prior Low  
- Compression / pause / coil  
- **In this app:** `D_Inside`, `W_Inside`, `M_Inside`

### **2U = Two-Up**
- High > prior High **AND** Low ≥ prior Low  
- Directional expansion up  
- **In this app:** shows up as Bull alignment (e.g. `D_Bull`, `W_Bull`, `M_Bull`)

### **2D = Two-Down**
- Low < prior Low **AND** High ≤ prior High  
- Directional expansion down  
- **In this app:** shows up as Bear alignment (e.g. `D_Bear`, `W_Bear`, `M_Bear`)

---

## 2) Actionable Triggers (What you actually place orders on)

### ✅ Inside Bar Break (Primary Trigger)
**LONG**
- **Entry:** Buy stop above the Inside Bar **High**
- **Stop:** Below the Inside Bar **Low**

**SHORT**
- **Entry:** Sell stop below the Inside Bar **Low**
- **Stop:** Above the Inside Bar **High**

**In this app:**
- We look for the **best trigger** in this order:
  1) **Weekly Inside Bar** (best)  
  2) **Daily Inside Bar**

You’ll see:  
- `D: READY / WAIT`  
- `W: READY / WAIT`  
- `M: INSIDE / —` (Monthly inside is noted but we don’t place monthly orders here)

---

## 3) The 2-1-2 Pattern (Continuation)

### **2-1-2 Up**
- 2U → 1 → 2U

### **2-1-2 Down**
- 2D → 1 → 2D

**In this app:**
- `D_212Up`, `W_212Up`
- `D_212Dn`, `W_212Dn`

**Simple use:**
- Treat it like compression → continuation in that direction.

---

## 4) Alignment (The “stacked timeframe” idea)

**Better odds when Weekly/Monthly agree.**

- LONG bias wants **Weekly Bull and/or Monthly Bull**
- SHORT bias wants **Weekly Bear and/or Monthly Bear**

**In this app:**
- Scanner bias is derived from the market ETFs (SPY/QQQ/IWM/DIA)
- Ticker analyzer can use market bias or manual bias

---

## 5) What “READY” means here (no fluff)

A ticker is **READY** when:
- It has a **Daily or Weekly Inside Bar**, AND
- The app can compute:
  - **Entry** and **Stop** from that bar

No inside bar = **WAIT** (and that’s correct)

---

## 6) Daily Workflow (2–5 minutes)

1) Open **Scanner**
2) Read **Market Bias**
3) Check **Rotation IN / Rotation OUT**
4) Drill into the strongest/weakest groups (based on bias)
5) Only take trades that show:
   - **W: READY** (best)
   - or **D: READY**
6) Place entry/stop orders and let it work
""")

    st.info("Next upgrade: we’ll add a STRAT-only ‘Trade Grade’ (alignment + trigger + room to target).")
