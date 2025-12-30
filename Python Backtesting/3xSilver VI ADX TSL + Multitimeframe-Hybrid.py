import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# --- Indikator-Funktionen ---

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
    
    tr = pd.concat([df['High']-df['Low'], 
                    abs(df['High']-df['Close'].shift(1)), 
                    abs(df['Low']-df['Close'].shift(1))], axis=1).max(axis=1)
    
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

def get_clean_returns(ticker):
    data = raw_data['Close'][ticker].ffill()
    rets = data.pct_change()
    # Der Schutz-Filter gegen extreme Daten-Ausreißer
    rets[abs(rets) > 0.5] = 0 
    return rets.fillna(0), data

returns_long, price_long = get_clean_returns(tickers["long_etf"])
returns_short, price_short = get_clean_returns(tickers["short_etf"])
daily_futures = raw_data.xs(tickers["futures"], axis=1, level=1).dropna()

# --- 2. Multi-Timeframe Signalberechnung ---
weekly_futures = daily_futures.resample('W-FRI').agg({'High':'max', 'Low':'min', 'Close':'last'})
w_vi_p, w_vi_m = calculate_vortex(weekly_futures, period=14)
weekly_trend = np.where(w_vi_p > w_vi_m, 1, -1)
weekly_trend_ser = pd.Series(weekly_trend, index=weekly_futures.index).shift(1)

d_vi_p, d_vi_m = calculate_vortex(daily_futures, period=14, ema_span=10)
d_adx = calculate_adx(daily_futures, period=14)

master_trend = weekly_trend_ser.reindex(daily_futures.index, method='ffill').fillna(0)

# --- 3. Simulation mit Trailing Stop-Loss Logik ---
tsl_percent = 0.15  # 15% Trailing Stop-Loss
capital = 10000.0
equity_curve = [capital]

current_pos = 0     # 0: Cash, 1: Long, -1: Short
peak_price = 0.0

for i in range(1, len(daily_futures)):
    # Signale vom Vortag (kein Look-Ahead)
    d_adx_val = d_adx.iloc[i-1]
    vi_cond_long = d_vi_p.iloc[i-1] > d_vi_m.iloc[i-1]
    vi_cond_short = d_vi_m.iloc[i-1] > d_vi_p.iloc[i-1]
    m_trend = master_trend.iloc[i-1]
    
    # Heutige Preise für TSL-Check
    p_long = price_long.iloc[i]
    p_short = price_short.iloc[i]
    
    # --- Positions-Management ---
    if current_pos == 0:
        # Einstieg suchen
        if d_adx_val > 25:
            if m_trend == 1 and vi_cond_long:
                current_pos = 1
                peak_price = p_long
            elif m_trend == -1 and vi_cond_short:
                current_pos = -1
                peak_price = p_short
                
    elif current_pos == 1:
        # Long-Management: Peak aktualisieren
        peak_price = max(peak_price, p_long)
        # Check Exit: TSL oder Signal-Trendwende
        if p_long < peak_price * (1 - tsl_percent) or not (m_trend == 1 and vi_cond_long):
            current_pos = 0
            
    elif current_pos == -1:
        # Short-Management (Short-ETF steigt wenn Silber fällt)
        peak_price = max(peak_price, p_short)
        # Check Exit: TSL oder Signal-Trendwende
        if p_short < peak_price * (1 - tsl_percent) or not (m_trend == -1 and vi_cond_short):
            current_pos = 0

    # Kapital-Berechnung
    ret = 0
    if current_pos == 1: ret = returns_long.iloc[i]
    elif current_pos == -1: ret = returns_short.iloc[i]
    
    capital *= (1 + ret)
    equity_curve.append(capital)

results = pd.DataFrame({'Equity': equity_curve}, index=daily_futures.index)

# --- 4. Visualisierung ---
plt.figure(figsize=(12,7))
plt.yscale('log')
plt.plot(results['Equity'], label=f'Hybrid + {int(tsl_percent*100)}% Trailing Stop', color='#d62728', linewidth=2)
plt.axhline(y=10000, color='black', linestyle='--', alpha=0.3, label='Startkapital')

ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"{int(x):,}".replace(",", ".")))
ax.yaxis.set_minor_formatter(mtick.NullFormatter())

plt.title('Silver 3x Strategy: Multi-Timeframe System with Trailing Stop-Loss', fontsize=14)
plt.grid(True, which="both", alpha=0.2)
plt.legend()
plt.show()

print(f"Finales Kapital mit TSL: {results['Equity'].iloc[-1]:,.2f} EUR".replace(",", "."))