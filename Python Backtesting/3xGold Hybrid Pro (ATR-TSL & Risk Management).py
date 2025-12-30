import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# =============================================================================
# HEADER: STRATEGIE-VARIABLEN (GOLD STABILITY STANDARD)
# =============================================================================
SYMBOL_FUTURES   = "GC=F"      # Gold Futures
SYMBOL_VOLA      = "^GVZ"      # CBOE Gold Volatility Index (Gold-VIX)
SYMBOL_LONG_ETF  = "3GOL.L"    # WisdomTree Gold 3x Daily Leveraged
SYMBOL_SHORT_ETF = "3GOS.L"    # WisdomTree Gold 3x Daily Short
START_DATE       = "2020-01-01"
INITIAL_CAPITAL  = 10000.0

# Indikator Parameter
VORTEX_PERIOD    = 14
EMA_SPAN_VORTEX  = 5           # Schnelle Glättung für Gold
ADX_PERIOD       = 14
ADX_ENTRY_LEVEL  = 20          # Sensitiver Einstieg
ADX_EXIT_LEVEL   = 18          # Hysterese-Exit
GVZ_CRITICAL     = 22.0        # Regime-Filter: Engerer Stop wenn GVZ > 22

# Risiko & Stop-Loss (Optimiert für 3x Gold)
ATR_MULT_STD     = 4.5         
ATR_MULT_TIGHT   = 2.5         
MAX_PERCENT_STOP = 0.12        # Hard-Cap: Maximal 12% vom Peak (ungehebelt)
HEBEL            = 3.0
MASTER_TREND_RES = "3D"        # Gold profitiert von 3rd-Day Resolution
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
data = yf.download([SYMBOL_FUTURES, SYMBOL_VOLA, SYMBOL_LONG_ETF, SYMBOL_SHORT_ETF], start=START_DATE, auto_adjust=True)

def clean_data(ticker):
    d = data['Close'][ticker].ffill()
    r = d.pct_change().fillna(0)
    r[abs(r) > 0.5] = 0 # Outlier Filter
    return r, d

rets_long, price_long = clean_data(SYMBOL_LONG_ETF)
rets_short, price_short = clean_data(SYMBOL_SHORT_ETF)
df_fut = data.xs(SYMBOL_FUTURES, axis=1, level=1).dropna()
df_gvz = data['Close'][SYMBOL_VOLA].ffill()

# --- 3. SIGNALBERECHNUNG ---
w_fut = df_fut.resample(MASTER_TREND_RES).agg({'High':'max', 'Low':'min', 'Close':'last'})
w_vi_p, w_vi_m = calculate_vortex(w_fut, VORTEX_PERIOD)
master_trend = (w_vi_p > w_vi_m).reindex(df_fut.index, method='ffill').shift(1).fillna(False)

d_vi_p, d_vi_m = calculate_vortex(df_fut, VORTEX_PERIOD, EMA_SPAN_VORTEX)
d_adx = calculate_adx(df_fut, ADX_PERIOD)
d_atr = pd.concat([df_fut['High']-df_fut['Low'], abs(df_fut['High']-df_fut['Close'].shift(1)), abs(df_fut['Low']-df_fut['Close'].shift(1))], axis=1).max(axis=1).rolling(14).mean()

# --- 4. SIMULATION MIT DAILY LAG & PROTECTION ---
cap = INITIAL_CAPITAL
equity = [cap]
pos = 0 
peak_price = 0.0
stop_level = 0.0 # Abendlich berechnetes Niveau für den nächsten Tag

for i in range(1, len(df_fut)):
    # Vorabend-Entscheidungswerte
    adx_v = d_adx.iloc[i-1]
    atr_v = d_atr.iloc[i-1]
    vola_v = df_gvz.iloc[i-1]
    m_trend = master_trend.iloc[i-1]
    vi_p, vi_m = d_vi_p.iloc[i-1], d_vi_m.iloc[i-1]
    
    # Heutige Intraday-Extrema
    curr_low = data['Low'][SYMBOL_LONG_ETF].iloc[i] if pos == 1 else data['Low'][SYMBOL_SHORT_ETF].iloc[i]
    curr_high = data['High'][SYMBOL_LONG_ETF].iloc[i] if pos == 1 else data['High'][SYMBOL_SHORT_ETF].iloc[i]
    
    # A. EXIT LOGIK (Check gegen Lag-Stop vom Vorabend)
    if pos != 0:
        if (pos == 1 and curr_low < stop_level) or (pos == -1 and curr_high > stop_level):
            pos = 0
        elif adx_v < ADX_EXIT_LEVEL or (pos == 1 and not (m_trend and vi_p > vi_m)) or (pos == -1 and not (not m_trend and vi_m > vi_p)):
            pos = 0

    # B. EINSTIEG LOGIK
    if pos == 0:
        if adx_v > ADX_ENTRY_LEVEL:
            if m_trend and vi_p > vi_m:
                pos, peak_price = 1, data['High'][SYMBOL_LONG_ETF].iloc[i]
            elif not m_trend and vi_m > vi_p:
                pos, peak_price = -1, data['Low'][SYMBOL_SHORT_ETF].iloc[i]
    
    # C. STOP-UPDATE FÜR NÄCHSTEN TAG (Abend-Berechnung)
    if pos != 0:
        if pos == 1: peak_price = max(peak_price, data['High'][SYMBOL_LONG_ETF].iloc[i])
        else: peak_price = min(peak_price, data['Low'][SYMBOL_SHORT_ETF].iloc[i])
        
        mult = ATR_MULT_TIGHT if (adx_v > 50 or vola_v > GVZ_CRITICAL) else ATR_MULT_STD
        # Dynamische Distanzberechnung unter Berücksichtigung des Hard-Caps
        tsl_dist_pct = min((atr_v * mult / df_fut['Close'].iloc[i]) * HEBEL, MAX_PERCENT_STOP * HEBEL)
        
        if pos == 1: stop_level = peak_price * (1 - tsl_dist_pct)
        else: stop_level = peak_price * (1 + tsl_dist_pct)

    # Kapital Update
    ret = (rets_long.iloc[i] if pos == 1 else rets_short.iloc[i] if pos == -1 else 0)
    cap *= (1 + ret)
    equity.append(cap)

# --- 5. VISUALISIERUNG ---
res = pd.DataFrame({'Equity': equity}, index=df_fut.index)
plt.figure(figsize=(14,8))
plt.yscale('log')
ax = plt.gca()

# Erweiterte Y-Achsen Ticks für logarithmische Darstellung
ax.yaxis.set_major_locator(mtick.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=20))
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', '.')))
ax.yaxis.set_minor_formatter(mtick.NullFormatter())

plt.plot(res['Equity'], label='Gold Hybrid Pro (Lag, GVZ, Hard-Cap)', color='#D4AF37', linewidth=2)
plt.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)

plt.title('Gold 3x Hybrid Strategy: Stability-Standard Backtest', fontsize=14)
plt.grid(True, which="major", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

final_cap_str = format(round(res['Equity'].iloc[-1], 2), ',').replace(',', 'X').replace('.', ',').replace('X', '.')
print(f"Finales Kapital Gold: {final_cap_str} EUR")