import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def calculate_vortex(df, period=14, ema_span=None):
    """Berechnet Vortex VI+ und VI- (optional mit EMA Glättung)"""
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

def calculate_adx(df, period=14):
    """Berechnet den ADX nach Wilder's Methode"""
    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']
    
    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))], axis=1).max(axis=1)
    tr_s = tr.ewm(alpha=1/period, adjust=False).mean()
    dm_p_s = pd.Series(dm_plus, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    dm_m_s = pd.Series(dm_minus, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    
    di_p = 100 * (dm_p_s / tr_s)
    di_m = 100 * (dm_m_s / tr_s)
    dx = 100 * abs(di_p - di_m) / (di_p + di_m)
    return dx.ewm(alpha=1/period, adjust=False).mean()

# --- 1. Daten Setup ---
tickers = {"futures": "SI=F", "long_etf": "3LSI.L", "short_etf": "3SSI.L"}
raw_data = yf.download(list(tickers.values()), start="2020-01-01", auto_adjust=True)

# Datenreinigung (Outlier Filter für Splits)
def get_clean_returns(ticker):
    data = raw_data['Close'][ticker].ffill()
    rets = data.pct_change()
    rets[abs(rets) > 0.5] = 0 # Split-Korrektur
    return rets.fillna(0)

returns_long = get_clean_returns(tickers["long_etf"])
returns_short = get_clean_returns(tickers["short_etf"])
daily_futures = raw_data.xs(tickers["futures"], axis=1, level=1).dropna()

# --- 2. Multi-Timeframe Signalberechnung ---
# A) Weekly Master Trend (Ohne EMA, um den "echten" Trend zu sehen)
weekly_futures = daily_futures.resample('W-FRI').agg({'High':'max', 'Low':'min', 'Close':'last'})
w_vi_p, w_vi_m = calculate_vortex(weekly_futures, period=14)
weekly_trend = np.where(w_vi_p > w_vi_m, 1, -1)
weekly_trend_ser = pd.Series(weekly_trend, index=weekly_futures.index).shift(1)

# B) Daily Execution Signal (Mit EMA 10 und ADX)
d_vi_p, d_vi_m = calculate_vortex(daily_futures, period=14, ema_span=10)
d_adx = calculate_adx(daily_futures, period=14)

# --- 3. Signal-Synchronisation & Logik ---
# Wir mappen den Weekly Trend auf die Daily Ebene
master_trend = weekly_trend_ser.reindex(daily_futures.index, method='ffill').fillna(0)

# Kombinierte Logik:
# 1 (Long): Weekly Trend ist POSITIV UND Daily VI+ > VI- UND ADX > 25
# -1 (Short): Weekly Trend ist NEGATIV UND Daily VI- > VI+ UND ADX > 25
# 0 (Cash): Sonst (Schutz vor Seitwärtsmarkt)
signal = np.zeros(len(daily_futures))
for i in range(len(daily_futures)):
    if d_adx.iloc[i] > 25:
        if master_trend.iloc[i] == 1 and d_vi_p.iloc[i] > d_vi_m.iloc[i]:
            signal[i] = 1
        elif master_trend.iloc[i] == -1 and d_vi_m.iloc[i] > d_vi_p.iloc[i]:
            signal[i] = -1

daily_signals = pd.Series(signal, index=daily_futures.index).shift(1).fillna(0)

# --- 4. Simulation ---
capital = 10000.0
equity_curve = [capital]

for i in range(1, len(daily_signals)):
    sig = daily_signals.iloc[i]
    ret = returns_long.iloc[i] if sig == 1 else (returns_short.iloc[i] if sig == -1 else 0)
    capital *= (1 + ret)
    equity_curve.append(capital)

results = pd.DataFrame({'Equity': equity_curve}, index=daily_signals.index)

# --- 5. Visualisierung ---
plt.figure(figsize=(12,7))
plt.yscale('log')
plt.plot(results['Equity'], label='Hybrid Strategy (Weekly Trend + Daily ADX/EMA)', color='#1f77b4', linewidth=2)
plt.axhline(y=10000, color='red', linestyle='--', alpha=0.5, label='Startkapital')

ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"{int(x):,}".replace(",", ".")))
ax.yaxis.set_minor_formatter(mtick.NullFormatter())

plt.title('Silver 3x Hybrid Strategy: Multi-Timeframe Protective System', fontsize=14)
plt.grid(True, which="both", alpha=0.2)
plt.legend()
plt.show()

print(f"Finales Kapital: {results['Equity'].iloc[-1]:.2f} EUR")