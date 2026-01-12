import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# =============================================================================
# HEADER: STRATEGIE-VARIABLEN (MIT VIX, HARD-CAP & DAILY LAG)
# =============================================================================
SYMBOL_FUTURES   = "ES=F"
SYMBOL_VIX       = "^VIX"
SYMBOL_LONG_ETF  = "3USL.L"
SYMBOL_SHORT_ETF = "3ULS.L"
START_DATE       = "2020-01-01"
INITIAL_CAPITAL  = 10000.0

# Indikator Parameter (Optimiert für SPX)
VORTEX_PERIOD    = 30 #20
EMA_SPAN_VORTEX  = 5    #7
ADX_PERIOD       = 14   #14
ADX_ENTRY_LEVEL  = 28     #23
ADX_EXIT_LEVEL   = 16      #17
VIX_CRITICAL     = 28.3564    #28.0    # Regime-Filter: Engerer Stop wenn VIX > 28

# Risiko & Stop-Loss
ATR_MULT_STD     = 4.44      #4.8
ATR_MULT_TIGHT   = 3.1      # 2.8
MAX_PERCENT_STOP = 0.12     #0.12   # Hard-Cap: Maximal 12% vom Peak (ungehebelt)
HEBEL            = 3.0     #3.0    # Hebel des ETFs
MASTER_TREND_RES = "W"
# =============================================================================

# --- 1. FUNKTIONEN ---
def calculate_vortex(df, period, ema_span=None):
    h_l = df['High'] - df['Low']
    tr = pd.concat([h_l, abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))], axis=1).max(axis=1)
    vmp = abs(df['High'] - df['Low'].shift(1))
    vmm = abs(df['Low'] - df['High'].shift(1))
    vi_plus = vmp.rolling(period).sum() / tr.rolling(period).sum()
    vi_minus = vmm.rolling(period).sum() / tr.rolling(period).sum()
    if ema_span:
        vi_plus = vi_plus.ewm(span=ema_span, adjust=False).mean()
        vi_minus = vi_minus.ewm(span=ema_span, adjust=False).mean()
    return vi_plus, vi_minus

def calculate_adx(df, period):
    plus_dm = (df['High'].diff()).clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(period).mean()

# --- 2. DATEN BESCHAFFUNG ---
data = yf.download([SYMBOL_FUTURES, SYMBOL_VIX, SYMBOL_LONG_ETF, SYMBOL_SHORT_ETF], start=START_DATE, auto_adjust=True)

def clean_data(ticker):
    d = data['Close'][ticker].ffill()
    r = d.pct_change().fillna(0)
    r[abs(r) > 0.5] = 0
    return r, d

rets_long, price_long = clean_data(SYMBOL_LONG_ETF)
rets_short, price_short = clean_data(SYMBOL_SHORT_ETF)
df_fut = data.xs(SYMBOL_FUTURES, axis=1, level=1).dropna()
df_vix = data['Close'][SYMBOL_VIX].ffill()

# --- 3. SIGNALBERECHNUNG ---
w_fut = df_fut.resample(MASTER_TREND_RES).agg({'High':'max', 'Low':'min', 'Close':'last'})
w_vi_p, w_vi_m = calculate_vortex(w_fut, VORTEX_PERIOD)
master_trend = (w_vi_p > w_vi_m).reindex(df_fut.index, method='ffill').shift(1).fillna(False)

d_vi_p, d_vi_m = calculate_vortex(df_fut, VORTEX_PERIOD, EMA_SPAN_VORTEX)
d_adx = calculate_adx(df_fut, ADX_PERIOD)
d_atr = pd.concat([df_fut['High']-df_fut['Low'], abs(df_fut['High']-df_fut['Close'].shift(1)), abs(df_fut['Low']-df_fut['Close'].shift(1))], axis=1).max(axis=1).rolling(14).mean()

# --- 4. SIMULATION MIT DAILY LAG ---
cap = INITIAL_CAPITAL
equity = [cap]
pos = 0
pos_size = 1.0
peak_price = 0.0
stop_level = 0.0 # Das "gepufferte" Stop-Level für den nächsten Tag

for i in range(1, len(df_fut)):
    # Werte vom Vorabend (i-1) für die Entscheidung an Tag (i)
    adx_v = d_adx.iloc[i-1]
    atr_v = d_atr.iloc[i-1]
    vix_v = df_vix.iloc[i-1]
    m_trend = master_trend.iloc[i-1]
    vi_p, vi_m = d_vi_p.iloc[i-1], d_vi_m.iloc[i-1]

    # Heutige Marktdaten (Intraday)
    curr_low = data['Low'][SYMBOL_LONG_ETF].iloc[i] if pos == 1 else data['Low'][SYMBOL_SHORT_ETF].iloc[i]
    curr_high = data['High'][SYMBOL_LONG_ETF].iloc[i] if pos == 1 else data['High'][SYMBOL_SHORT_ETF].iloc[i]

    # A. EXIT LOGIK (Check gegen das Stop-Level vom VORABEND)
    if pos != 0:
        # TSL-Bruch Check
        if (pos == 1 and curr_low < stop_level) or (pos == -1 and curr_high > stop_level):
            pos = 0
        # Signal-Bruch Check (Trendwende am Abend erkannt)
        elif adx_v < ADX_EXIT_LEVEL or (pos == 1 and not (m_trend and vi_p > vi_m)) or (pos == -1 and not (not m_trend and vi_m > vi_p)):
            pos = 0

    # B. EINSTIEG LOGIK
    if pos == 0:
        if adx_v > ADX_ENTRY_LEVEL:
            if m_trend and vi_p > vi_m:
                pos, peak_price = 1, data['High'][SYMBOL_LONG_ETF].iloc[i]
            elif not m_trend and vi_m > vi_p:
                pos, peak_price = -1, data['Low'][SYMBOL_SHORT_ETF].iloc[i]

    # C. STOP-LEVEL UPDATE FÜR DEN NÄCHSTEN TAG (Dein "Abend-Ritual")
    if pos != 0:
        # Bestimme Peak
        if pos == 1: peak_price = max(peak_price, data['High'][SYMBOL_LONG_ETF].iloc[i])
        else: peak_price = min(peak_price, data['Low'][SYMBOL_SHORT_ETF].iloc[i])

        # Berechne neuen dynamischen Stop-Abstand
        mult = ATR_MULT_TIGHT if (adx_v > 50 or vix_v > VIX_CRITICAL) else ATR_MULT_STD
        tsl_dist_pct = min((atr_v * mult / df_fut['Close'].iloc[i]) * HEBEL, MAX_PERCENT_STOP * HEBEL)

        if pos == 1: stop_level = peak_price * (1 - tsl_dist_pct)
        else: stop_level = peak_price * (1 + tsl_dist_pct)

    # Kapital Update
    ret = (rets_long.iloc[i] if pos == 1 else rets_short.iloc[i] if pos == -1 else 0)
    cap *= (1 + ret)
    equity.append(cap)

# --- 5. PLOT ---
res = pd.DataFrame({'Equity': equity}, index=df_fut.index)
plt.figure(figsize=(14, 8)) # Etwas breiter für bessere Lesbarkeit
plt.yscale('log')

ax = plt.gca()

# Definition der Ticks: Zeige Ticks bei 1, 2, 5 mal der Zehnerpotenz
# Das generiert Marken bei 10k, 20k, 50k, 100k, 200k, 500k...
y_locator = mtick.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=20)
ax.yaxis.set_major_locator(y_locator)

# Formatter für deutsche Tausender-Punkte
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', '.')))

# Hilfslinien für alle Ticks aktivieren
plt.grid(True, which="major", color='gray', linestyle='-', alpha=0.3)
plt.grid(True, which="minor", color='gray', linestyle=':', alpha=0.1)

plt.plot(res['Equity'], label='S&P 500 Hybrid Pro (Vortex 20)', color='#00aaff', linewidth=2)
plt.axhline(y=INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.5, label="Initial Capital")

plt.title(f'Advanced Backtest S&P 500 3x Leveraged', fontsize=14)
plt.ylabel('Kapital in EUR', fontsize=12)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()