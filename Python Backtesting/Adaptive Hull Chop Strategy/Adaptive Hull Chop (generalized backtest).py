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

# Wähle hier das Asset aus (muss dem Dateinamen entsprechen: 'silver', 'gold', 'sp500')
SELECTED_ASSET = "sp500" 

def load_config(asset_name):
    """Lädt die Konfiguration aus einer externen JSON-Datei."""
    filename = f"{asset_name.lower()}.json"
    
    if not os.path.exists(filename):
        print(f"KRITISCHER FEHLER: Konfigurationsdatei '{filename}' nicht gefunden!")
        print(f"Bitte erstelle '{filename}' im Skript-Verzeichnis.")
        sys.exit(1)
        
    try:
        with open(filename, 'r') as f:
            config = json.load(f)
            print(f"Konfiguration geladen: {filename} -> {config['NAME']}")
            return config
    except json.JSONDecodeError as e:
        print(f"FEHLER: '{filename}' ist kein gültiges JSON. {e}")
        sys.exit(1)

# Config laden
cfg = load_config(SELECTED_ASSET)
START_DATE = "2020-01-01"
INITIAL_CAPITAL = 10000.0

# =============================================================================
# 2. HILFSFUNKTIONEN (MATHEMATIK)
# =============================================================================

def calculate_wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_hma(series, length):
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = calculate_wma(series, half_length)
    wma_full = calculate_wma(series, length)
    raw_hma = 2 * wma_half - wma_full
    return calculate_wma(raw_hma, sqrt_length)

def calculate_chop(df, length):
    # True Range für 1 Periode
    tr1 = pd.concat([df['High'] - df['Low'], 
                     abs(df['High'] - df['Close'].shift(1)), 
                     abs(df['Low'] - df['Close'].shift(1))], axis=1).max(axis=1)
    
    atr_sum = tr1.rolling(length).sum()
    high_max = df['High'].rolling(length).max()
    low_min = df['Low'].rolling(length).min()
    
    # Division durch Null verhindern
    denominator = high_max - low_min
    denominator = denominator.replace(0, 0.000001) 
    
    numerator = np.log10(atr_sum / denominator)
    denom_log = np.log10(length)
    
    return 100 * (numerator / denom_log)

def calculate_rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

# =============================================================================
# 3. DATENLADEN & VORBEREITUNG
# =============================================================================
print("Lade Marktdaten...")
tickers = [cfg["SYMBOL_FUTURES"], cfg["SYMBOL_VOLA"], cfg["SYMBOL_LONG_ETF"], cfg["SYMBOL_SHORT_ETF"]]

try:
    data = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False)
except Exception as e:
    print(f"Fehler beim Download: {e}")
    sys.exit(1)

try:
    # Zugriff über Config-Keys
    df_fut = data.xs(cfg["SYMBOL_FUTURES"], axis=1, level=1).dropna()
    df_vola = data.xs(cfg["SYMBOL_VOLA"], axis=1, level=1)['Close'].ffill()
    price_long = data.xs(cfg["SYMBOL_LONG_ETF"], axis=1, level=1)['Close'].ffill()
    price_short = data.xs(cfg["SYMBOL_SHORT_ETF"], axis=1, level=1)['Close'].ffill()
    
    rets_long = price_long.pct_change().fillna(0)
    rets_short = price_short.pct_change().fillna(0)
    
    high_long = data.xs(cfg["SYMBOL_LONG_ETF"], axis=1, level=1)['High'].ffill()
    low_long = data.xs(cfg["SYMBOL_LONG_ETF"], axis=1, level=1)['Low'].ffill()
    high_short = data.xs(cfg["SYMBOL_SHORT_ETF"], axis=1, level=1)['High'].ffill()
    low_short = data.xs(cfg["SYMBOL_SHORT_ETF"], axis=1, level=1)['Low'].ffill()

except KeyError as e:
    print(f"Datenstruktur Fehler: {e}. Bitte prüfen, ob die Ticker im JSON korrekt sind.")
    sys.exit(1)

# =============================================================================
# 4. INDIKATOREN BERECHNUNG
# =============================================================================
print("Berechne Indikatoren...")

# HMA
df_fut['HMA'] = calculate_hma(df_fut['Close'], cfg["HMA_LENGTH"])
df_fut['HMA_Rising'] = df_fut['HMA'] > df_fut['HMA'].shift(1)

# Chop
df_fut['CHOP'] = calculate_chop(df_fut, cfg["CHOP_LENGTH"])

# ATR
tr = pd.concat([df_fut['High'] - df_fut['Low'], 
                abs(df_fut['High'] - df_fut['Close'].shift(1)), 
                abs(df_fut['Low'] - df_fut['Close'].shift(1))], axis=1).max(axis=1)
df_fut['ATR'] = calculate_rma(tr, cfg["ATR_LENGTH"])

# Vola Align
df_vola = df_vola.reindex(df_fut.index).ffill()

# =============================================================================
# 5. BACKTEST SIMULATION (State Machine)
# =============================================================================
print("Starte Simulation...")

cap = INITIAL_CAPITAL
equity = [cap]
pos = 0          
peak = 0.0       
stop_lvl = 0.0   
locked = False   
last_dir = 0     

for i in range(1, len(df_fut)):
    # --- A. PREVIOUS DAY DATA ---
    close_prev = df_fut['Close'].iloc[i-1]
    hma_prev = df_fut['HMA'].iloc[i-1]
    chop_prev = df_fut['CHOP'].iloc[i-1]
    atr_prev = df_fut['ATR'].iloc[i-1]
    hma_rising = df_fut['HMA_Rising'].iloc[i-1]
    vol_val = df_vola.iloc[i-1]
    
    # --- B. CURRENT DAY DATA ---
    p_l_low = low_long.iloc[i]
    p_s_high = high_short.iloc[i]
    pl_ret = rets_long.iloc[i]
    ps_ret = rets_short.iloc[i]

    # 1. UNLOCK CHECK
    if pos == 0 and locked:
        tech_reset = False
        if last_dir == 1 and not hma_rising: tech_reset = True
        if last_dir == -1 and hma_rising: tech_reset = True
        
        if tech_reset:
            locked = False

    # 2. EXIT LOGIC
    exit_triggered = False
    
    if pos == 1:
        if p_l_low < stop_lvl:
            pos = 0; locked = True; last_dir = 1; exit_triggered = True
        elif close_prev < hma_prev: 
            pos = 0; exit_triggered = True 

    elif pos == -1:
        if p_s_high > stop_lvl:
            pos = 0; locked = True; last_dir = -1; exit_triggered = True
        elif close_prev > hma_prev:
            pos = 0; exit_triggered = True

    # 3. ENTRY LOGIC
    if pos == 0 and not locked:
        is_trending_cond = chop_prev < cfg["CHOP_THRESHOLD"]
        
        if is_trending_cond:
            if close_prev > hma_prev and hma_rising:
                pos = 1
                peak = high_long.iloc[i]
                stop_lvl = 0
                
            elif close_prev < hma_prev and not hma_rising:
                pos = -1
                peak = low_short.iloc[i]
                stop_lvl = 999999

    # 4. TSL UPDATE (Config Based)
    if pos != 0:
        is_stress = (vol_val > cfg["VOLA_CRITICAL"]) or (chop_prev > 60)
        mult = cfg["ATR_MULT_TIGHT"] if is_stress else cfg["ATR_MULT_STD"]
        
        dist_pct = (atr_prev * mult / close_prev) * cfg["HEBEL"]
        final_pct = min(dist_pct, cfg["MAX_PERCENT_STOP"] * cfg["HEBEL"])
        
        if pos == 1:
            curr_high = high_long.iloc[i]
            peak = max(peak, curr_high)
            new_stop = peak * (1 - final_pct)
            stop_lvl = max(stop_lvl, new_stop)
            
        elif pos == -1:
            curr_low = low_short.iloc[i]
            peak = min(peak, curr_low)
            new_stop = peak * (1 + final_pct)
            if stop_lvl == 999999: stop_lvl = new_stop
            else: stop_lvl = min(stop_lvl, new_stop)

    # 5. CAPITAL UPDATE
    if pos == 1: cap *= (1 + pl_ret)
    elif pos == -1: cap *= (1 + ps_ret)
    equity.append(cap)

# =============================================================================
# 6. VISUALISIERUNG
# =============================================================================
res = pd.DataFrame({'Equity': equity}, index=df_fut.index)
final_return = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

plt.figure(figsize=(14, 8))
plt.yscale('log')
ax = plt.gca()

y_locator = mtick.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=20)
ax.yaxis.set_major_locator(y_locator)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', '.')))

plt.grid(True, which="major", color='gray', linestyle='-', alpha=0.3)
plt.grid(True, which="minor", color='gray', linestyle=':', alpha=0.1)

plt.plot(res['Equity'], label=f'{cfg["NAME"]} (3x)', color='#2962FF', linewidth=2)
plt.axhline(y=INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.5, label="Startkapital")

plt.title(f'Backtest: {cfg["NAME"]} | Final: {final_return:.2f}%', fontsize=14)
plt.ylabel('Kapital in EUR', fontsize=12)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

print(f"Endkapital: {cap:.2f} EUR")