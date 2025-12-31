import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# =============================================================================
# HEADER: STRATEGIE-VARIABLEN (SILVER STABILITY STANDARD)
# =============================================================================
SYMBOL_FUTURES   = "SI=F"      # Silber Futures
SYMBOL_VOLA      = "^VXSLV"    # CBOE Silver ETF Volatility Index
SYMBOL_LONG_ETF  = "3LSI.L"    # WisdomTree Silver 3x Daily Leveraged
SYMBOL_SHORT_ETF = "3SSI.L"    # WisdomTree Silver 3x Daily Short
START_DATE       = "2020-01-01"
INITIAL_CAPITAL  = 10000.0

# Indikator Parameter (Silver Needs Speed & Room)
VORTEX_PERIOD    = 14
EMA_SPAN_VORTEX  = 10          # Stärkere Glättung für Silber-Rauschen
ADX_PERIOD       = 14
ADX_ENTRY_LEVEL  = 25          # Hohe Hürde, um "Fake-Ausbrüche" zu vermeiden
ADX_EXIT_LEVEL   = 20          # Engere Hysterese für Silber
VXSLV_CRITICAL   = 38.0        # Regime-Filter: Engerer Stop wenn VXSLV > 38

# Risiko & Stop-Loss (Hard-Cap Protection)
ATR_MULT_STD     = 3.5         # Silber braucht weniger Platz als Gold, da explosiver
ATR_MULT_TIGHT   = 2.0         
MAX_PERCENT_STOP = 0.12        # Hard-Cap: 12% vom Peak (ungehebelt)
HEBEL            = 3.0
MASTER_TREND_RES = "3D"        # 3-Tage-Auflösung ist der Sweet-Spot für Silber
# =============================================================================

def calculate_vortex(df, period, ema_span=None):
    h_l, h_pc, l_pc = df['High']-df['Low'], abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    vmp, vmm = abs(df['High']-df['Low'].shift(1)), abs(df['Low']-df['High'].shift(1))
    vi_plus = vmp.rolling(period).sum() / tr.rolling(period).sum()
    vi_minus = vmm.rolling(period).sum() / tr.rolling(period).sum()
    if ema_span:
        vi_plus = vi_plus.ewm(span=ema_span, adjust=False).mean()
        vi_minus = vi_minus.ewm(span=ema_span, adjust=False).mean()
    return vi_plus, vi_minus

def calculate_adx(df, period):
    plus_dm, minus_dm = (df['High'].diff()).clip(lower=0), (-df['Low'].diff()).clip(lower=0)
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    p_di, m_di = 100*(plus_dm.rolling(period).mean()/atr), 100*(minus_dm.rolling(period).mean()/atr)
    dx = 100 * abs(p_di - m_di) / (p_di + m_di)
    return dx.rolling(period).mean()

# --- DATEN & SIMULATION ---
data = yf.download([SYMBOL_FUTURES, SYMBOL_VOLA, SYMBOL_LONG_ETF, SYMBOL_SHORT_ETF], start=START_DATE, auto_adjust=True)
rets_long, price_long = data['Close'][SYMBOL_LONG_ETF].ffill().pct_change().fillna(0), data['Close'][SYMBOL_LONG_ETF].ffill()
rets_short, price_short = data['Close'][SYMBOL_SHORT_ETF].ffill().pct_change().fillna(0), data['Close'][SYMBOL_SHORT_ETF].ffill()
df_fut = data.xs(SYMBOL_FUTURES, axis=1, level=1).dropna()
df_vola = data['Close'][SYMBOL_VOLA].ffill()

# Signal-Basis
w_fut = df_fut.resample(MASTER_TREND_RES).agg({'High':'max', 'Low':'min', 'Close':'last'})
w_vi_p, w_vi_m = calculate_vortex(w_fut, VORTEX_PERIOD)
m_trend = (w_vi_p > w_vi_m).reindex(df_fut.index, method='ffill').shift(1).fillna(False)
d_vi_p, d_vi_m = calculate_vortex(df_fut, VORTEX_PERIOD, EMA_SPAN_VORTEX)
d_adx, d_atr = calculate_adx(df_fut, ADX_PERIOD), pd.concat([df_fut['High']-df_fut['Low'], abs(df_fut['High']-df_fut['Close'].shift(1)), abs(df_fut['Low']-df_fut['Close'].shift(1))], axis=1).max(axis=1).rolling(14).mean()

cap = INITIAL_CAPITAL
equity, pos, peak, stop_lvl = [cap], 0, 0.0, 0.0

for i in range(1, len(df_fut)):
    adx_v, atr_v, vol_v, mt, vp, vm = d_adx.iloc[i-1], d_atr.iloc[i-1], df_vola.iloc[i-1], m_trend.iloc[i-1], d_vi_p.iloc[i-1], d_vi_m.iloc[i-1]
    c_low = data['Low'][SYMBOL_LONG_ETF].iloc[i] if pos==1 else data['Low'][SYMBOL_SHORT_ETF].iloc[i]
    c_high = data['High'][SYMBOL_LONG_ETF].iloc[i] if pos==1 else data['High'][SYMBOL_SHORT_ETF].iloc[i]

    if pos != 0: # EXIT CHECK (Lagged)
        if (pos == 1 and c_low < stop_lvl) or (pos == -1 and c_high > stop_lvl) or adx_v < ADX_EXIT_LEVEL: pos = 0
        elif not ((pos==1 and mt and vp>vm) or (pos==-1 and not mt and vm>vp)): pos = 0
    
    if pos == 0 and adx_v > ADX_ENTRY_LEVEL: # ENTRY
        if mt and vp > vm: pos, peak = 1, data['High'][SYMBOL_LONG_ETF].iloc[i]
        elif not mt and vm > vp: pos, peak = -1, data['Low'][SYMBOL_SHORT_ETF].iloc[i]

    if pos != 0: # ABEND-UPDATE (Stop für morgen)
        peak = max(peak, data['High'][SYMBOL_LONG_ETF].iloc[i]) if pos==1 else min(peak, data['Low'][SYMBOL_SHORT_ETF].iloc[i])
        mult = ATR_MULT_TIGHT if (adx_v > 50 or vol_v > VXSLV_CRITICAL) else ATR_MULT_STD
        tsl_p = min((atr_v * mult / df_fut['Close'].iloc[i]) * HEBEL, MAX_PERCENT_STOP * HEBEL)
        stop_lvl = peak * (1 - tsl_p) if pos==1 else peak * (1 + tsl_p)

    cap *= (1 + (rets_long.iloc[i] if pos==1 else rets_short.iloc[i] if pos==-1 else 0))
    equity.append(cap)

# --- PLOT ---
# --- 5. VISUALISIERUNG MIT ERWEITERTER Y-ACHSE ---
res = pd.DataFrame({'Equity': equity}, index=df_fut.index)
plt.figure(figsize=(14, 8))
plt.yscale('log')

ax = plt.gca()

# 1. Locator: Bestimmt, WO die Striche sitzen. 
# subs=(1, 2, 5) erzeugt Ticks bei 10k, 20k, 50k, 100k, 200k, 500k...
y_locator = mtick.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=20)
ax.yaxis.set_major_locator(y_locator)

# 2. Formatter: Bestimmt, WIE die Zahlen aussehen (Deutsch: Punkt als Tausender-Trenner)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', '.')))

# 3. Grid: Major-Grid für die beschrifteten Werte, Minor-Grid für die Optik
plt.grid(True, which="major", color='gray', linestyle='-', alpha=0.3)
plt.grid(True, which="minor", color='gray', linestyle=':', alpha=0.1)

# Plotten der Kurve
plt.plot(res['Equity'], label='Hybrid Pro Strategy (Optimized)', color='#D4AF37', linewidth=2)
plt.axhline(y=INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.5, label="Startkapital")

plt.title('Advanced Backtest: Equity Curve with Detailed Y-Axis', fontsize=14)
plt.ylabel('Kapital in EUR', fontsize=12)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()