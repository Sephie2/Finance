import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# =============================================================================
# HEADER: STRATEGIE-VARIABLEN (LOG-MACD EDITION MIT VIX & HARD-CAP)
# =============================================================================
SYMBOL_FUTURES   = "ES=F"      
SYMBOL_VIX       = "^VIX"       
SYMBOL_LONG_ETF  = "3USL.L"    
SYMBOL_SHORT_ETF = "3ULS.L"    
START_DATE       = "2020-01-01"
INITIAL_CAPITAL  = 10000.0

# MACD Parameter (Log-Raum)
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9

# Volatilitäts-Regime
VIX_CRITICAL     = 28.0        

# Risiko & Stop-Loss (Exakt wie im Basis-Script)
ATR_MULT_STD     = 4.8         
ATR_MULT_TIGHT   = 2.8         
MAX_PERCENT_STOP = 0.12        
HEBEL            = 3.0
# =============================================================================

# --- 1. FUNKTIONEN ---
def calculate_macd_log(df, fast=12, slow=26, signal=9):
    """Berechnet den MACD auf Basis des natürlichen Logarithmus."""
    log_price = np.log(df['Close'])
    exp1 = log_price.ewm(span=fast, adjust=False).mean()
    exp2 = log_price.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calculate_atr(df, period=14):
    h_l = df['High'] - df['Low']
    tr = pd.concat([h_l, abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))], axis=1).max(axis=1)
    return tr.rolling(period).mean()

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

# --- 3. SIGNALBERECHNUNG (LOG-MACD) ---
d_macd, d_signal = calculate_macd_log(df_fut, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
d_atr = calculate_atr(df_fut, 14)

# --- 4. SIMULATION MIT DAILY LAG & MOMENTUM-ENTRY ---
cap = INITIAL_CAPITAL
equity = [cap]
pos = 0 
peak_price = 0.0
stop_level = 0.0 

for i in range(1, len(df_fut)):
    # Werte vom Vorabend (i-1)
    macd_v = d_macd.iloc[i-1]
    sig_v  = d_signal.iloc[i-1]
    # Für Crossover-Erkennung:
    macd_v_prev = d_macd.iloc[i-2] if i > 1 else macd_v
    sig_v_prev  = d_signal.iloc[i-2] if i > 1 else sig_v
    
    atr_v = d_atr.iloc[i-1]
    vix_v = df_vix.iloc[i-1]
    
    # Heutige Marktdaten (Intraday)
    curr_low = data['Low'][SYMBOL_LONG_ETF].iloc[i] if pos == 1 else data['Low'][SYMBOL_SHORT_ETF].iloc[i]
    curr_high = data['High'][SYMBOL_LONG_ETF].iloc[i] if pos == 1 else data['High'][SYMBOL_SHORT_ETF].iloc[i]
    
    # A. EXIT LOGIK (Check gegen TSL vom Vorabend + MACD Trendbruch)
    if pos != 0:
        # 1. Trailing Stop Loss Trigger
        tsl_triggered = (pos == 1 and curr_low < stop_level) or (pos == -1 and curr_high > stop_level)
        
        # 2. MACD Momentum-Exit (Crossover in Gegenrichtung am Vorabend)
        momentum_exit = (pos == 1 and macd_v < sig_v) or (pos == -1 and macd_v > sig_v)
        
        if tsl_triggered or momentum_exit:
            pos = 0

    # B. EINSTIEG LOGIK (MACD Crossover am Vorabend)
    if pos == 0:
        # Long Entry: MACD kreuzt Signal nach oben
        if macd_v > sig_v and macd_v_prev <= sig_v_prev:
            pos, peak_price = 1, data['High'][SYMBOL_LONG_ETF].iloc[i]
        # Short Entry: MACD kreuzt Signal nach unten
        elif macd_v < sig_v and macd_v_prev >= sig_v_prev:
            pos, peak_price = -1, data['Low'][SYMBOL_SHORT_ETF].iloc[i]
    
    # C. STOP-LEVEL UPDATE FÜR DEN NÄCHSTEN TAG (Abend-Ritual)
    if pos != 0:
        if pos == 1: peak_price = max(peak_price, data['High'][SYMBOL_LONG_ETF].iloc[i])
        else: peak_price = min(peak_price, data['Low'][SYMBOL_SHORT_ETF].iloc[i])
        
        # Volatilitäts-Regime Check
        mult = ATR_MULT_TIGHT if vix_v > VIX_CRITICAL else ATR_MULT_STD
        tsl_dist_pct = min((atr_v * mult / df_fut['Close'].iloc[i]) * HEBEL, MAX_PERCENT_STOP * HEBEL)
        
        if pos == 1: stop_level = peak_price * (1 - tsl_dist_pct)
        else: stop_level = peak_price * (1 + tsl_dist_pct)

    # Kapital Update
    ret = (rets_long.iloc[i] if pos == 1 else rets_short.iloc[i] if pos == -1 else 0)
    cap *= (1 + ret)
    equity.append(cap)

# --- 5. PLOT ---
res = pd.DataFrame({'Equity': equity}, index=df_fut.index)
plt.figure(figsize=(14, 8))
plt.yscale('log')
ax = plt.gca()
y_locator = mtick.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=20)
ax.yaxis.set_major_locator(y_locator)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', '.')))

plt.grid(True, which="major", color='gray', linestyle='-', alpha=0.3)
plt.plot(res['Equity'], label='S&P 500 Log-MACD Hybrid', color='#ff9900', linewidth=2)
plt.axhline(y=INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.5, label="Startkapital")

plt.title('Backtest: S&P 500 3x Leveraged (Log-MACD + ATR-TSL)', fontsize=14)
plt.ylabel('Kapital in EUR', fontsize=12)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()