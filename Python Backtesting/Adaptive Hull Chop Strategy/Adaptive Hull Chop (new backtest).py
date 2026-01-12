import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import json
import os
import sys

# =============================================================================
# 1. KONFIGURATION & LOADING
# =============================================================================
SELECTED_ASSET = "silver" 

def load_config(asset_name):
    filename = f"{asset_name.lower()}.json"
    if not os.path.exists(filename):
        print(f"KRITISCHER FEHLER: '{filename}' nicht gefunden!")
        sys.exit(1)
    with open(filename, 'r') as f:
        return json.load(f)

cfg = load_config(SELECTED_ASSET)
START_DATE = "2020-01-01"
INITIAL_CAPITAL = 10000.0

# =============================================================================
# 2. HILFSFUNKTIONEN
# =============================================================================
def calculate_wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_hma(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    return calculate_wma(2 * calculate_wma(series, half_length) - calculate_wma(series, length), sqrt_length)

def calculate_chop(df, length):
    tr1 = pd.concat([df['High'] - df['Low'], abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))], axis=1).max(axis=1)
    denom = (df['High'].rolling(length).max() - df['Low'].rolling(length).min()).replace(0, 0.000001)
    return 100 * (np.log10(tr1.rolling(length).sum() / denom) / np.log10(length))

def calculate_rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

# =============================================================================
# 3. BACKTEST
# =============================================================================
print(f"Lade Marktdaten für {cfg['SYMBOL_FUTURES']}...")
tickers = [cfg["SYMBOL_FUTURES"], cfg["SYMBOL_VOLA"], cfg["SYMBOL_LONG_ETF"], cfg["SYMBOL_SHORT_ETF"]]
try:
    data = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False)
except:
    print("Download Fehler")
    sys.exit(1)

df_fut = data.xs(cfg["SYMBOL_FUTURES"], axis=1, level=1).dropna()
df_vola = data.xs(cfg["SYMBOL_VOLA"], axis=1, level=1)['Close'].ffill().reindex(df_fut.index).ffill()
price_long = data.xs(cfg["SYMBOL_LONG_ETF"], axis=1, level=1)['Close'].ffill()
price_short = data.xs(cfg["SYMBOL_SHORT_ETF"], axis=1, level=1)['Close'].ffill()
high_long = data.xs(cfg["SYMBOL_LONG_ETF"], axis=1, level=1)['High'].ffill()
low_long = data.xs(cfg["SYMBOL_LONG_ETF"], axis=1, level=1)['Low'].ffill()
high_short = data.xs(cfg["SYMBOL_SHORT_ETF"], axis=1, level=1)['High'].ffill()
low_short = data.xs(cfg["SYMBOL_SHORT_ETF"], axis=1, level=1)['Low'].ffill()

# Indikatoren
df_fut['HMA'] = calculate_hma(df_fut['Close'], cfg["HMA_LENGTH"])
df_fut['HMA_Rising'] = df_fut['HMA'] > df_fut['HMA'].shift(1)
df_fut['CHOP'] = calculate_chop(df_fut, cfg["CHOP_LENGTH"])
tr = pd.concat([df_fut['High'] - df_fut['Low'], abs(df_fut['High'] - df_fut['Close'].shift(1)), abs(df_fut['Low'] - df_fut['Close'].shift(1))], axis=1).max(axis=1)
df_fut['ATR'] = calculate_rma(tr, cfg["ATR_LENGTH"])

# Simulation
cap = INITIAL_CAPITAL
equity = [cap]
pos = 0; peak = 0.0; stop_lvl = 0.0; locked = False; last_dir = 0

# --- NEU: STATISTIK VARIABLEN ---
trade_count = 0
winning_trades = 0
losing_trades = 0
entry_price = 0.0

print("Starte Simulation...")

for i in range(1, len(df_fut)):
    c_prev = df_fut['Close'].iloc[i-1]
    hma_prev = df_fut['HMA'].iloc[i-1]
    chop_prev = df_fut['CHOP'].iloc[i-1]
    atr_prev = df_fut['ATR'].iloc[i-1]
    hma_rising = df_fut['HMA_Rising'].iloc[i-1]
    vol_val = df_vola.iloc[i-1]
    
    # 1. UNLOCK
    if pos == 0 and locked:
        tech_reset = False
        if last_dir == 1 and not hma_rising: tech_reset = True
        if last_dir == -1 and hma_rising: tech_reset = True
        if tech_reset: locked = False

    # 2. EXIT
    exit_triggered = False
    trade_pnl = 0.0
    
    if pos != 0:
        old_pos = pos
        # Check Exits
        if pos == 1:
            if low_long.iloc[i] < stop_lvl:
                pos = 0; locked = True; last_dir = 1
            elif c_prev < hma_prev: 
                pos = 0
        elif pos == -1:
            if high_short.iloc[i] > stop_lvl:
                pos = 0; locked = True; last_dir = -1
            elif c_prev > hma_prev:
                pos = 0
        
        # Wenn Position geschlossen wurde -> Stats update
        if old_pos != 0 and pos == 0:
            # Einfache PnL Schätzung basierend auf Kapitalveränderung seit Entry wäre genauer, 
            # hier nutzen wir einfach: Ist aktuelles Kapital > Kapital beim Entry?
            # Da wir Cap täglich updaten, ist das schwerer im Loop.
            # Wir prüfen einfach den Daily Return des Exit-Tages (vereinfacht)
            pass 

    # 3. ENTRY
    if pos == 0 and not locked:
        is_trending_cond = chop_prev < cfg["CHOP_THRESHOLD"]
        if is_trending_cond:
            if c_prev > hma_prev and hma_rising:
                pos = 1; peak = high_long.iloc[i]; stop_lvl = 0
                trade_count += 1
            elif c_prev < hma_prev and not hma_rising:
                pos = -1; peak = low_short.iloc[i]; stop_lvl = 999999
                trade_count += 1

    # 4. TSL UPDATE
    if pos != 0:
        is_stress = (vol_val > cfg["VOLA_CRITICAL"]) or (chop_prev > 60)
        mult = cfg["ATR_MULT_TIGHT"] if is_stress else cfg["ATR_MULT_STD"]
        dist_pct = (atr_prev * mult / c_prev) * cfg["HEBEL"]
        final_pct = min(dist_pct, cfg["MAX_PERCENT_STOP"] * cfg["HEBEL"])
        
        if pos == 1:
            peak = max(peak, high_long.iloc[i])
            stop_lvl = max(stop_lvl, peak * (1 - final_pct))
        elif pos == -1:
            peak = min(peak, low_short.iloc[i])
            stop_lvl = min(stop_lvl, min(stop_lvl, peak * (1 + final_pct))) if stop_lvl != 999999 else peak * (1 + final_pct)

    # 5. CAPITAL
    r = 0.0
    if pos == 1: r = price_long.pct_change().fillna(0).iloc[i]
    elif pos == -1: r = price_short.pct_change().fillna(0).iloc[i]
    cap *= (1 + r)
    equity.append(cap)

# --- ANALYSE & DRAWDOWN ---
equity_curve = pd.Series(equity, index=df_fut.index)
running_max = equity_curve.cummax()
drawdown = (equity_curve - running_max) / running_max
max_dd = drawdown.min() * 100
final_return = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

print("-" * 40)
print(f"ERGEBNIS ANALYSE: {cfg['NAME']}")
print("-" * 40)
print(f"Endkapital:      {cap:,.2f} EUR")
print(f"Gesamt-Return:   {final_return:.2f}%")
print(f"Anzahl Trades:   {trade_count}")  # <--- HIER GENAU HINSCHAUEN
print(f"Max Drawdown:    {max_dd:.2f}%")
print("-" * 40)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(equity_curve, label='Equity', color='#2962FF')
ax1.set_yscale('log')
ax1.grid(True, which='both', alpha=0.3)
ax1.set_title(f"Equity Curve (Trades: {trade_count})")
ax1.legend()

ax2.plot(drawdown * 100, label='Drawdown %', color='red', linewidth=1)
ax2.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
ax2.set_title("Drawdown")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()