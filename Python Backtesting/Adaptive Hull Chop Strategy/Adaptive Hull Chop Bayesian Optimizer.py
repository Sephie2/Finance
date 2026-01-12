import yfinance as yf
import pandas as pd
import numpy as np
import optuna
import logging
import json
import os
import sys

# Optuna Logging reduzieren
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# 1. KONFIGURATION & LOADING
# =============================================================================

# Wähle das Asset, das optimiert werden soll (muss als .json existieren)
SELECTED_ASSET = "sp500" 
TRIAL_RUNS = 100  # Anzahl der Versuche

def load_config(asset_name):
    filename = f"{asset_name.lower()}.json"
    if not os.path.exists(filename):
        print(f"KRITISCHER FEHLER: '{filename}' nicht gefunden!")
        sys.exit(1)
    try:
        with open(filename, 'r') as f:
            config = json.load(f)
            print(f"--- Optimiere Strategie: {config['NAME']} ---")
            return config
    except Exception as e:
        print(f"JSON Fehler: {e}")
        sys.exit(1)

cfg = load_config(SELECTED_ASSET)
START_DATE = "2020-01-01"

# =============================================================================
# 2. DATEN LADEN (Dynamisch aus Config)
# =============================================================================
print(f"Lade Marktdaten für {cfg['SYMBOL_FUTURES']}...")
tickers = [cfg["SYMBOL_FUTURES"], cfg["SYMBOL_VOLA"], cfg["SYMBOL_LONG_ETF"], cfg["SYMBOL_SHORT_ETF"]]

try:
    raw_data = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False)
except Exception as e:
    print(f"Download Fehler: {e}")
    sys.exit(1)

try:
    # Daten zuweisen
    df_fut = raw_data.xs(cfg["SYMBOL_FUTURES"], axis=1, level=1).dropna()
    df_vola = raw_data.xs(cfg["SYMBOL_VOLA"], axis=1, level=1)['Close'].ffill()
    
    # ETFs
    price_long = raw_data.xs(cfg["SYMBOL_LONG_ETF"], axis=1, level=1)['Close'].ffill()
    price_short = raw_data.xs(cfg["SYMBOL_SHORT_ETF"], axis=1, level=1)['Close'].ffill()
    
    rets_long = price_long.pct_change().fillna(0)
    rets_short = price_short.pct_change().fillna(0)
    
    high_long = raw_data.xs(cfg["SYMBOL_LONG_ETF"], axis=1, level=1)['High'].ffill()
    low_long = raw_data.xs(cfg["SYMBOL_LONG_ETF"], axis=1, level=1)['Low'].ffill()
    high_short = raw_data.xs(cfg["SYMBOL_SHORT_ETF"], axis=1, level=1)['High'].ffill()
    low_short = raw_data.xs(cfg["SYMBOL_SHORT_ETF"], axis=1, level=1)['Low'].ffill()

except KeyError as e:
    print(f"Datenstruktur Fehler: {e}. Prüfe die Ticker in {SELECTED_ASSET}.json")
    sys.exit(1)

# =============================================================================
# 3. HELPER FUNCTIONS (Vektorisierte Indikatoren)
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
    tr = pd.concat([df['High'] - df['Low'], 
                    abs(df['High'] - df['Close'].shift(1)), 
                    abs(df['Low'] - df['Close'].shift(1))], axis=1).max(axis=1)
    atr_sum = tr.rolling(length).sum()
    high_max = df['High'].rolling(length).max()
    low_min = df['Low'].rolling(length).min()
    
    denom = high_max - low_min
    denom = denom.replace(0, 0.0001)
    
    numerator = np.log10(atr_sum / denom)
    denominator = np.log10(length)
    return 100 * (numerator / denominator)

def calculate_rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

# =============================================================================
# 4. OBJECTIVE FUNCTION
# =============================================================================
def objective(trial):
    # A. SUCHRAUM (Search Space)
    # Wir suchen "um" die Standardwerte herum, aber geben genug Freiheit
    hma_len     = trial.suggest_int("HMA_LENGTH", 30, 80)
    chop_len    = trial.suggest_int("CHOP_LENGTH", 10, 20)
    chop_thresh = trial.suggest_float("CHOP_THRESHOLD", 35.0, 60.0)
    
    atr_std     = trial.suggest_float("ATR_MULT_STD", 3.0, 7.0)
    atr_tight   = trial.suggest_float("ATR_MULT_TIGHT", 1.5, 4.0)
    # Vola Trigger: Range etwas an das Asset anpassen (Gold ist niedriger, SP500 höher)
    # Wir nehmen +/- 10 Punkte um den aktuellen Wert in der Config
    v_base = cfg.get("VOLA_CRITICAL", 25.0)
    vola_crit   = trial.suggest_float("VOLA_CRITICAL", max(10.0, v_base - 10), v_base + 15)
    
    # B. INDIKATOREN VORBERECHNEN (Speed)
    hma = calculate_hma(df_fut['Close'], hma_len)
    chop = calculate_chop(df_fut, chop_len)
    
    # ATR (RMA based, fix 14)
    tr = pd.concat([df_fut['High'] - df_fut['Low'], 
                    abs(df_fut['High'] - df_fut['Close'].shift(1)), 
                    abs(df_fut['Low'] - df_fut['Close'].shift(1))], axis=1).max(axis=1)
    atr = calculate_rma(tr, 14)
    
    hma_rising = hma > hma.shift(1)
    
    # C. SIMULATION LOOP (State Machine)
    cap = 10000.0
    equity = [cap]
    pos = 0; peak = 0.0; stop_lvl = 0.0; locked = False; last_dir = 0
    hebel = cfg["HEBEL"] # Hebel aus Config laden
    
    # Numpy Arrays für Speed
    close_arr = df_fut['Close'].values
    hma_arr = hma.values
    chop_arr = chop.values
    atr_arr = atr.values
    hma_rise_arr = hma_rising.values
    vola_arr = df_vola.reindex(df_fut.index).ffill().values
    
    ret_l_arr = rets_long.values
    ret_s_arr = rets_short.values
    high_l_arr = high_long.values
    low_l_arr = low_long.values
    high_s_arr = high_short.values
    low_s_arr = low_short.values
    
    # Loop
    for i in range(hma_len + 1, len(df_fut)):
        # Daten
        c_prev = close_arr[i-1]
        hma_prev = hma_arr[i-1]
        chop_prev = chop_arr[i-1]
        atr_prev = atr_arr[i-1]
        rise_prev = hma_rise_arr[i-1]
        vol_prev = vola_arr[i-1]
        
        p_l_low = low_l_arr[i]
        p_s_high = high_s_arr[i]
        
        # 1. Unlock
        if pos == 0 and locked:
            tech_reset = False
            if last_dir == 1 and not rise_prev: tech_reset = True
            if last_dir == -1 and rise_prev: tech_reset = True
            if tech_reset: locked = False
                
        # 2. Exit
        if pos == 1:
            if p_l_low < stop_lvl:
                pos = 0; locked = True; last_dir = 1
            elif c_prev < hma_prev:
                pos = 0
        elif pos == -1:
            if p_s_high > stop_lvl:
                pos = 0; locked = True; last_dir = -1
            elif c_prev > hma_prev:
                pos = 0
                
        # 3. Entry
        if pos == 0 and not locked:
            is_trending = chop_prev < chop_thresh
            if is_trending:
                if c_prev > hma_prev and rise_prev:
                    pos = 1; peak = high_l_arr[i]; stop_lvl = 0
                elif c_prev < hma_prev and not rise_prev:
                    pos = -1; peak = low_s_arr[i]; stop_lvl = 999999
                    
        # 4. TSL Update
        if pos != 0:
            is_stress = (vol_prev > vola_crit) or (chop_prev > 60)
            mult = atr_tight if is_stress else atr_std
            tsl_dist = (atr_prev * mult / c_prev) * hebel
            final_pct = min(tsl_dist, cfg["MAX_PERCENT_STOP"] * hebel)
            
            if pos == 1:
                peak = max(peak, high_l_arr[i])
                stop_lvl = max(stop_lvl, peak * (1 - final_pct))
            elif pos == -1:
                peak = min(peak, low_s_arr[i])
                new_stop = peak * (1 + final_pct)
                if stop_lvl == 999999: stop_lvl = new_stop
                else: stop_lvl = min(stop_lvl, new_stop)
                
        # 5. PnL
        r = 0.0
        if pos == 1: r = ret_l_arr[i]
        elif pos == -1: r = ret_s_arr[i]
        cap *= (1 + r)
        equity.append(cap)
        
    # Metrics
    eq_ser = pd.Series(equity)
    returns = eq_ser.pct_change().fillna(0)
    if returns.std() == 0: return -1
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    
    trial.set_user_attr("final_cap", cap)
    return sharpe

# =============================================================================
# 5. START OPTIMIZATION
# =============================================================================
print(f"Starte Optimierung ({TRIAL_RUNS} Runs)...")
study = optuna.create_study(direction="maximize")

# WICHTIG: Wir injizieren die Werte aus der JSON als ersten Versuch!
# Das garantiert, dass der Optimizer weiß, was der aktuelle Stand ist.
initial_params = {
    "HMA_LENGTH": cfg["HMA_LENGTH"],
    "CHOP_LENGTH": cfg["CHOP_LENGTH"],
    "CHOP_THRESHOLD": cfg["CHOP_THRESHOLD"],
    "ATR_MULT_STD": cfg["ATR_MULT_STD"],
    "ATR_MULT_TIGHT": cfg["ATR_MULT_TIGHT"],
    "VOLA_CRITICAL": cfg["VOLA_CRITICAL"]
}
print(f"Startwerte aus JSON: {initial_params}")
study.enqueue_trial(initial_params)

study.optimize(objective, n_trials=TRIAL_RUNS)

# =============================================================================
# 6. OUTPUT & JSON UPDATE STRING
# =============================================================================
print("-" * 50)
print(f"OPTIMIERUNG ABGESCHLOSSEN: {cfg['NAME']}")
print(f"Beste Sharpe Ratio: {study.best_value:.4f}")
print(f"Maximales Endkapital: {study.best_trial.user_attrs['final_cap']:.2f} EUR")
print("-" * 50)

# Neue Config zum Kopieren vorbereiten
new_config = cfg.copy()
# Update mit besten Parametern
for k, v in study.best_params.items():
    # Optuna gibt floats manchmal mit extrem vielen Nachkommastellen zurück -> Runden
    if isinstance(v, float):
        new_config[k] = round(v, 2)
    else:
        new_config[k] = v

print("Kopiere diesen Block in deine JSON-Datei, um die neuen Werte zu speichern:")
print("-" * 20)
print(json.dumps(new_config, indent=4))
print("-" * 20)