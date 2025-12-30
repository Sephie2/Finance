import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def calculate_indicators(df, vortex_period=14, adx_period=14, ema_span=10):
    """Berechnet Vortex (geglättet) und ADX auf Tagesbasis"""
    # --- Vortex Indicator ---
    h_l = df['High'] - df['Low']
    h_pc = abs(df['High'] - df['Close'].shift(1))
    l_pc = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    
    vmp = abs(df['High'] - df['Low'].shift(1))
    vmm = abs(df['Low'] - df['High'].shift(1))
    
    vi_plus = vmp.rolling(vortex_period).sum() / tr.rolling(vortex_period).sum()
    vi_minus = vmm.rolling(vortex_period).sum() / tr.rolling(vortex_period).sum()
    
    # EMA Glättung auf Vortex (EMA 10)
    vi_plus_ema = vi_plus.ewm(span=ema_span, adjust=False).mean()
    vi_minus_ema = vi_minus.ewm(span=ema_span, adjust=False).mean()

    # --- ADX (Directional Movement Index) ---
    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']
    
    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_smooth = tr.ewm(alpha=1/adx_period, adjust=False).mean()
    dm_plus_smooth = pd.Series(dm_plus, index=df.index).ewm(alpha=1/adx_period, adjust=False).mean()
    dm_minus_smooth = pd.Series(dm_minus, index=df.index).ewm(alpha=1/adx_period, adjust=False).mean()
    
    di_plus = 100 * (dm_plus_smooth / tr_smooth)
    di_minus = 100 * (dm_minus_smooth / tr_smooth)
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.ewm(alpha=1/adx_period, adjust=False).mean()
    
    return vi_plus_ema, vi_minus_ema, adx

# --- 1. Datenbeschaffung & Reinigung ---
tickers = {"futures": "SI=F", "long_etf": "3LSI.L", "short_etf": "3SSI.L"}
raw_data = yf.download(list(tickers.values()), start="2020-01-01", auto_adjust=True)

def clean_returns(series):
    returns = series.pct_change()
    returns[abs(returns) > 0.5] = 0 
    return returns.fillna(0)

returns_long = clean_returns(raw_data['Close'][tickers["long_etf"]])
returns_short = clean_returns(raw_data['Close'][tickers["short_etf"]])

# --- 2. Signalberechnung auf DAILY Basis ---
# Zugriff auf die Futures-Daten ohne wöchentliches Resampling
daily_futures = raw_data.xs(tickers["futures"], axis=1, level=1).dropna()

vi_p, vi_m, adx = calculate_indicators(daily_futures)

# Strategie-Bedingungen (Daily Check)
trend_strength_threshold = 25
signal = np.where((adx > trend_strength_threshold), 
                  np.where(vi_p > vi_m, 1, -1), 
                  0)

# Shift(1): Signal von heute Abend wird morgen gehandelt
daily_signals = pd.Series(signal, index=daily_futures.index).shift(1).fillna(0)

# Synchronisierung der Index-Zeitstempel
daily_signals = daily_signals.reindex(returns_long.index, method='ffill').fillna(0)

# --- 3. Backtest Simulation ---
capital = 10000.0
equity_curve = [capital]

for i in range(1, len(daily_signals)):
    sig = daily_signals.iloc[i]
    if sig == 1:
        ret = returns_long.iloc[i]
    elif sig == -1:
        ret = returns_short.iloc[i]
    else:
        ret = 0 
    
    capital *= (1 + ret)
    equity_curve.append(capital)

results = pd.DataFrame({'Equity': equity_curve}, index=daily_signals.index)

# --- 4. Visualisierung ---
plt.figure(figsize=(12,7))
plt.yscale('log')
plt.plot(results['Equity'], label='Daily Vortex EMA10 + ADX Strategie', color='#2ca02c', linewidth=2)
plt.axhline(y=10000, color='red', linestyle='--', alpha=0.5, label='Startkapital')

def plain_num(x, pos): return f"{int(x):,}".replace(",", ".")
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(plain_num))
plt.gca().yaxis.set_minor_formatter(mtick.NullFormatter())

plt.title('Silver 3x Strategy: DAILY EMA-Smoothed Vortex & ADX Filter', fontsize=14)
plt.grid(True, which="both", alpha=0.2)
plt.legend()
plt.show()

print(f"Finales Kapital: {results['Equity'].iloc[-1]:.2f}")