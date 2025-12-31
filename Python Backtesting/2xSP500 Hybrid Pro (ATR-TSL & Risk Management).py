import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# =============================================================================
# HEADER: STRATEGIE-VARIABLEN (ZUM EXPERIMENTIEREN)
# =============================================================================
SYMBOL_FUTURES   = "ES=F"      # S&P 500 E-mini Futures
SYMBOL_LONG_ETF  = "DBPG.DE"    # WisdomTree S&P 500 3x Daily Leveraged "3USL.L" 
SYMBOL_SHORT_ETF = "2B7A.PA"    # WisdomTree S&P 500 3x Daily Short "3ULS.L"
START_DATE       = "2020-01-01"
INITIAL_CAPITAL  = 10000.0

# Indikator Parameter
VORTEX_PERIOD    = 20  ## 20
EMA_SPAN_VORTEX  = 7 ## 7          # Glättung für Daily Vortex
ADX_PERIOD       = 14
ADX_ENTRY_LEVEL  = 23 ##23          # Einstieg bei Trendstärke
ADX_EXIT_LEVEL   = 17 ## 17          # Exit bei Trendschwäche (Hysterese)
ADX_EXTREME      = 50          # Grenze für Risiko-Management

# Risiko & Stop-Loss
ATR_PERIOD       = 14
ATR_MULT_STD     = 4.8  ## 4.8      # Standard Puffer
ATR_MULT_TIGHT   = 2.8  ## 2.8         # Enger Puffer bei ADX > 50
HEBEL            = 2.0
MASTER_TREND_RES = "W"         # "W" für Weekly, "3D" für 3 Tage
# =============================================================================

# --- 1. FUNKTIONEN ---
def calculate_vortex(df, period, ema_span=None):
    h_l = df['High'] - df['Low']
    h_pc = abs(df['High'] - df['Close'].shift(1))
    l_pc = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    vmp = abs(df['High'] - df['Low'].shift(1))
    vmm = abs(df['Low'] - df['High'].shift(1))
    vi_plus = vmp.rolling(period).sum() / tr.rolling(period).sum()
    vi_minus = vmm.rolling(period).sum() / tr.rolling(period).sum()
    if ema_span:
        vi_plus = vi_plus.ewm(span=ema_span, adjust=False).mean()
        vi_minus = vi_minus.ewm(span=ema_span, adjust=False).mean()
    return vi_plus, vi_minus

def calculate_adx(df, period):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff().apply(lambda x: -x)
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))], axis=1).max(axis=1)
    atr_smooth = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_smooth)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(period).mean()

def calculate_atr(df, period):
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# --- 2. DATEN BESCHAFFUNG ---
data = yf.download([SYMBOL_FUTURES, SYMBOL_LONG_ETF, SYMBOL_SHORT_ETF], start=START_DATE, auto_adjust=True)

def clean_data(ticker):
    d = data['Close'][ticker].ffill()
    r = d.pct_change().fillna(0)
    r[abs(r) > 0.5] = 0 
    return r, d

rets_long, price_long = clean_data(SYMBOL_LONG_ETF)
rets_short, price_short = clean_data(SYMBOL_SHORT_ETF)
df_fut = data.xs(SYMBOL_FUTURES, axis=1, level=1).dropna()

# --- 3. SIGNALBERECHNUNG ---
w_fut = df_fut.resample(MASTER_TREND_RES).agg({'High':'max', 'Low':'min', 'Close':'last'})
w_vi_p, w_vi_m = calculate_vortex(w_fut, VORTEX_PERIOD)
master_trend = (w_vi_p > w_vi_m).reindex(df_fut.index, method='ffill').shift(1).fillna(False)

d_vi_p, d_vi_m = calculate_vortex(df_fut, VORTEX_PERIOD, EMA_SPAN_VORTEX)
d_adx = calculate_adx(df_fut, ADX_PERIOD)
d_atr = calculate_atr(df_fut, ATR_PERIOD)

# --- 4. SIMULATION ---
cap = INITIAL_CAPITAL
equity = [cap]
pos = 0 # 0: Cash, 1: Long, -1: Short
pos_size = 1.0
peak = 0.0
is_halved = False

for i in range(1, len(df_fut)):
    adx_v = d_adx.iloc[i-1]
    atr_v = d_atr.iloc[i-1]
    m_trend = master_trend.iloc[i-1]
    vi_p, vi_m = d_vi_p.iloc[i-1], d_vi_m.iloc[i-1]
    
    p_l, p_s = price_long.iloc[i], price_short.iloc[i]
    r_l, r_s = rets_long.iloc[i], rets_short.iloc[i]
    
    # Dynamischer TSL
    mult = ATR_MULT_TIGHT if adx_v > ADX_EXTREME else ATR_MULT_STD
    tsl = (atr_v * mult / df_fut['Close'].iloc[i-1]) * HEBEL
    
    if pos == 0:
        is_halved = False
        pos_size = 1.0
        if adx_v > ADX_ENTRY_LEVEL:
            if m_trend and vi_p > vi_m:
                pos, peak = 1, p_l
            elif not m_trend and vi_m > vi_p:
                pos, peak = -1, p_s
                
    elif pos == 1:
        if adx_v > ADX_EXTREME and not is_halved:
            pos_size, is_halved = 0.5, True
        peak = max(peak, p_l)
        if p_l < peak * (1 - tsl) or adx_v < ADX_EXIT_LEVEL or not (m_trend and vi_p > vi_m):
            pos = 0
            
    elif pos == -1:
        if adx_v > ADX_EXTREME and not is_halved:
            pos_size, is_halved = 0.5, True
        peak = max(peak, p_s)
        if p_s < peak * (1 - tsl) or adx_v < ADX_EXIT_LEVEL or not (not m_trend and vi_m > vi_p):
            pos = 0

    cur_ret = (r_l if pos == 1 else r_s if pos == -1 else 0) * pos_size
    cap *= (1 + cur_ret)
    equity.append(cap)

# --- 5. PLOT MIT ERWEITERTER Y-ACHSE ---
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