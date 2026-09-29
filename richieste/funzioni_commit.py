
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import statsmodels.api as sm
from matplotlib import pyplot as plt
from richieste.logiche_predict import (
    _run_negbin_logic,
    _run_ols_logic,
    _run_negbin_sarimax_logic)

from richieste.logiche_create import (
    create_negbin_sarimax_model,
    create_negbin_model,
    create_ols_model,
    create_gaussian_glm_model
)

#==================================================================================================================================================================

def compute_metrics(df, pred_col="y_pred", real_col="y_real"):
    resid = df[real_col] - df[pred_col]
    mae  = resid.abs().mean()
    rmse = np.sqrt((resid**2).mean())
    mape = (resid.abs() / df[real_col].replace(0, np.nan)).mean()
    mse  = (resid**2).mean()
    r2 = r2_score(df[real_col], df[pred_col])
    bias = np.mean(resid)
    return mae, rmse, mape, mse, r2, bias

#==================================================================================================================================================================

def create_dummy_features(df, column, prefix, drop_first=True):
    """
    Crea variabili dummy per una colonna categorica.
    
    Parameters:
    -----------
    df : DataFrame
    column : str, nome colonna
    prefix : str, prefisso dummy
    drop_first : bool, se True usa baseline encoding
    
    Returns:
    --------
    df_features : DataFrame con le feature create
    feature_names : list, nomi colonne create
    """
    dummies = pd.get_dummies(df[column], prefix=prefix, drop_first=drop_first)
    df_features = dummies.astype(float)
    
    return df_features, list(dummies.columns)

#==================================================================================================================================================================

def create_fourier_features(df, period, n_harmonics, prefix):
    """
    Crea armoniche di Fourier per una data periodicità.
    
    Parameters:
    -----------
    df : DataFrame
    period : float, periodicità (es. 24 per giornaliero)
    n_harmonics : int, numero armoniche da creare
    prefix : str, prefisso nomi colonne (es. 'hour')
    
    Returns:
    --------
    df_features : DataFrame con le feature create
    feature_names : list, nomi colonne create
    """
    df_features = pd.DataFrame(index=df.index)
    feature_names = []
    
    for k in range(1, n_harmonics + 1):
        # Sin
        col_sin = f'{prefix}_sin{k}'
        df_features[col_sin] = np.sin(2 * np.pi * k * df[f'{prefix}'] / period)
        feature_names.append(col_sin)
        
        # Cos
        col_cos = f'{prefix}_cos{k}'
        df_features[col_cos] = np.cos(2 * np.pi * k * df[f'{prefix}'] / period)
        feature_names.append(col_cos)
    
    return df_features, feature_names

#==================================================================================================================================================================

def create_all_base_features(df, k_hour=6, k_week=4):
    """
    Crea tutte le feature base per il modello.
    
    Feature create:
    - Armoniche orarie (k_hour)
    - Armoniche settimanali (k_week) 
    - Dummy giorni settimana (6, baseline=Dom)
    - Dummy mesi (11, baseline=Dic)
    - Indicator holiday
    
    Parameters:
    -----------
    df : DataFrame originale
    k_hour : int, numero armoniche orarie
    k_week : int, numero armoniche settimanali
    
    Returns:
    --------
    df_features : DataFrame con tutte le feature create
    feature_names : list, tutti i nomi feature create
    """
    print("Creando feature base...")
    
    # Inizializza DataFrame per le feature
    df_features = pd.DataFrame(index=df.index)
    features = []
    
    # 1. Armoniche ORARIE (periodo 24h)
    # print(f"  • {k_hour} armoniche orarie (periodo 24h)...")
    hour_df, hour_features = create_fourier_features(df, period=24, n_harmonics=k_hour, 
                                                      prefix='hour_corrected')
    df_features = pd.concat([df_features, hour_df], axis=1)
    features.extend(hour_features)
    
    # 2. Armoniche SETTIMANALI (periodo 168h = 7 giorni × 24 ore)
    # print(f"  • {k_week} armoniche settimanali (periodo 168h)...")
    # Crea hour_of_week temporaneo: 0 (Lun 00:00) → 167 (Dom 23:00)
    df_temp = df.copy()
    df_temp['hour_of_week'] = df['day_of_week'] * 24 + df['hour_corrected']
    week_df, week_features = create_fourier_features(df_temp, period=168, n_harmonics=k_week,
                                                      prefix='hour_of_week')
    df_features = pd.concat([df_features, week_df], axis=1)
    features.extend(week_features)
    
    # 3. Dummy GIORNI SETTIMANA (baseline = Domenica)
    # print(f"  • Dummy giorni settimana (baseline=Domenica)...")
    day_df, day_features = create_dummy_features(df, 'day_of_week', prefix='day', drop_first=True)
    df_features = pd.concat([df_features, day_df], axis=1)
    features.extend(day_features)
    
    # 4. Dummy MESI (baseline = Dicembre)
    # print(f"  • Dummy mesi (baseline=Dicembre)...")
    month_df, month_features = create_dummy_features(df, 'month', prefix='month', drop_first=True)
    df_features = pd.concat([df_features, month_df], axis=1)
    features.extend(month_features)
    
    # 5. Indicator HOLIDAY
    # print(f"  • Indicator holiday...")
    df_features['is_holiday_float'] = df['is_holiday'].astype(float)
    features.append('is_holiday_float')
    
    # 6. Indicator Rush hour
    # print(f"  • Indicator holiday...")
    df_features['is_rush_hour'] = df['is_rush_hour'].astype(float)
    features.append('is_rush_hour')
    print(f"\n✓ {len(features)} feature base create")
    
    return df_features, features

#==================================================================================================================================================================

def create_interactions(df_features, base_features):
    """
    Crea interazioni chiave tra feature.
    
    Interazioni create:
    1. Giorno × Armoniche Ora (cattura: pattern orario diverso per ogni giorno)
    2. Mese × Armoniche Ora (cattura: Agosto diverso da altri mesi)
    3. Holiday × Armoniche Ora (cattura: picco spostato nei festivi)
    
    Parameters:
    -----------
    df_features : DataFrame con le feature base
    base_features : list, nomi delle feature base
    
    Returns:
    --------
    df_interactions : DataFrame con le interazioni create
    interaction_features : list, nomi feature interazioni
    """
    df_interactions = pd.DataFrame(index=df_features.index)
    interactions = []
    
    # Identifica feature
    day_cols = [f for f in base_features if f.startswith('day_')]
    month_cols = [f for f in base_features if f.startswith('month_')]
    hour_harm_cols = [f for f in base_features if 'hour_corrected' in f]
    
    # 1. GIORNO × ARMONICHE ORA
    # print("Creando interazioni giorno × armoniche ora...")
    n_interactions = 0
    for day_col in day_cols:
        for hour_col in hour_harm_cols:
            interact_name = f'{day_col}_{hour_col}'
            df_interactions[interact_name] = df_features[day_col] * df_features[hour_col]
            interactions.append(interact_name)
            n_interactions += 1
    
    # print(f"  ✓ {n_interactions} interazioni giorno × ora")
    # n_interactions = 0
    # for day_col in day_cols:
    #     for month_col in month_cols:
    #         interact_name = f'{day_col}_{month_col}'
    #         df_interactions[interact_name] = df_features[day_col] * df_features[month_col]
    #         interactions.append(interact_name)
    #         n_interactions += 1
    
    # print(f"  ✓ {n_interactions} interazioni giorno × ora")
    
    # 2. MESE × ARMONICHE ORA
    # print("Creando interazioni mese × armoniche ora...")
    n_interactions = 0
    for month_col in month_cols:
        for hour_col in hour_harm_cols:
            interact_name = f'{month_col}_{hour_col}'
            df_interactions[interact_name] = df_features[month_col] * df_features[hour_col]
            interactions.append(interact_name)
            n_interactions += 1
    
    # print(f"  ✓ {n_interactions} interazioni mese × ora")
    
    # 3. HOLIDAY × ARMONICHE ORA
    # print("Creando interazioni holiday × armoniche ora...")
    n_interactions = 0
    for hour_col in hour_harm_cols:
        interact_name = f'holiday_{hour_col}'
        df_interactions[interact_name] = df_features['is_holiday_float'] * df_features[hour_col]
        interactions.append(interact_name)
        n_interactions += 1
    #"  ✓ {n_interactions} interazioni holiday × ora") print(f
    
    # 4. RUSH × ARMONICHE ORA
    # print("Creando interazioni holiday × armoniche ora...")
    n_interactions = 0
    for hour_col in hour_harm_cols:
        interact_name = f'rush_{hour_col}'
        df_interactions[interact_name] = df_features['is_rush_hour'] * df_features[hour_col]
        interactions.append(interact_name)
        n_interactions += 1
    # print(f"  ✓ {n_interactions} interazioni rush hour × ora")
    
    print(f"\n✓ Totale {len(interactions)} interazioni create\n")
    
    return df_interactions, interactions

#==================================================================================================================================================================

def apply_dst_correction(df):

    def is_dst_italy(dt):
        year = dt.year

        march_31 = datetime(year, 3, 31)
        days_to_sunday = (march_31.weekday() + 1) % 7
        dst_start = (march_31 - timedelta(days=days_to_sunday)).replace(
            hour=2, minute=0, second=0
        )

        oct_31 = datetime(year, 10, 31)
        days_to_sunday = (oct_31.weekday() + 1) % 7
        dst_end = (oct_31 - timedelta(days=days_to_sunday)).replace(
            hour=3, minute=0, second=0
        )

        return dst_start <= dt < dst_end

    df['is_dst'] = df['datetime'].apply(is_dst_italy)
    df['hour_sensor'] = df['datetime'].dt.hour
    df['hour_corrected'] = df['hour_sensor']

    df.loc[df['is_dst'], 'hour_corrected'] = (
        df.loc[df['is_dst'], 'hour_corrected'] + 1
    ) % 24

    return df

#==================================================================================================================================================================

def ensure_const_and_align(X_train, X_test, const_name="const"):
    X_train = X_train.copy()
    X_test  = X_test.copy()

    # 1. Aggiunta esplicita della costante se manca
    if const_name not in X_train.columns:
        X_train.insert(0, const_name, 1.0)

    if const_name not in X_test.columns:
        X_test.insert(0, const_name, 1.0)

    # 2. Allineamento colonne (ordine e presenza)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    # 3. Controlli finali (fail fast)
    assert X_train.shape[1] == X_test.shape[1], \
        f"Mismatch colonne: train={X_train.shape[1]}, test={X_test.shape[1]}"

    assert list(X_train.columns) == list(X_test.columns), \
        "Le colonne non sono allineate"

    return X_train, X_test

#==================================================================================================================================================================

def run_rolling_evaluation(df, y, X, horizon, periods, model_type='OLS'):
    """
    Funzione universale per la valutazione rolling.
    Modelli supportati: 'OLS' (OLS Log + SARIMAX), 'NegBin', 'Prophet'
    """
    test_size = horizon * periods
    train_size = len(df) - test_size

    # Split dati
    y_train, y_test = y.iloc[:train_size], y.iloc[-test_size:]
    X_train, X_test = X.iloc[:train_size, :], X.iloc[-test_size:, :]
    ds_test = df['datetime'].iloc[-test_size:]


    # Aggiungo costante
    X_train_const, X_test_const = ensure_const_and_align(X_train, X_test)
    # --- LOGICA DISCRIMINANTE ---
    
    if model_type.upper() == 'OLS':
        results_df = _run_ols_logic(y_train, y_test, X_train_const, X_test_const, ds_test)
        
    elif model_type.upper() == 'NEGBIN':
        results_df = _run_negbin_logic(y_train, y_test, X_train_const, X_test_const, ds_test, test_size)
    
    elif model_type.upper() == 'PROPHET':
        # Prepariamo il dataframe di train includendo le esogene
        train_prophet = X_train_const.copy()
        train_prophet['ds'] = df['datetime'].iloc[:train_size].values
        train_prophet['y'] = y_train.values
  
    else:
        raise ValueError(f"Modello {model_type} non supportato.")

    # Calcolo metriche comuni per il print (scala lineare)
    r2 = r2_score(results_df['y_real'], results_df['y_pred'])
    rmse = np.sqrt(mean_squared_error(results_df['y_real'], results_df['y_pred']))
    mae = mean_absolute_error(results_df['y_real'], results_df['y_pred'])
    
    print("="*70)
    print(f"REPORT ROLLING - MODELLO: {model_type.upper()}")
    print("="*70)
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    
    return results_df
#==================================================================================================================================================================
def format_results(results_df):
    results_df['y_pred']    = np.expm1(results_df['y_pred_log_corrected'])
    results_df['residuals'] = results_df['y_pred'] - results_df['y_real']

    print(f"\n✓ Calcolate previsioni e residui in scala lineare")
    results_df = results_df.copy()
    z = 2.33  # ~99%

    results_df["upper_99_log"] = (
        results_df["y_pred_log_corrected"] +z * results_df["sigma_garch"]
    )
    results_df["lower_99_log"] = (
        results_df["y_pred_log_corrected"] -z * results_df["sigma_garch"]
    )

    results_df["lower_99"] = np.expm1(results_df['lower_99_log'])
    results_df["upper_99"] = np.expm1(results_df['upper_99_log'])

    print(f"\n✓ Bande di confidenza create correttamente")
    return results_df
#==================================================================================================================================================================
def create_final_df(df):
    final_df = df[['ds', 'y_real', 'lower_99', 'y_pred', 'upper_99', 'is_anomaly']]
    final_df["true_anomaly"] = (
    final_df["is_anomaly"]
    & ((final_df["lower_99"] - final_df["y_real"]) / (final_df["y_pred"] + 1e-6) > 0.2)
    & (final_df['lower_99'] > 0.05*final_df["y_real"].mean())
)
    print('il numero assoluto di anomalie possibili è ' + str(final_df['is_anomaly'].sum()))
    print('il numero di anomalie possibili, in percentuale, è ' + str(final_df['is_anomaly'].mean() * 100) + '%' + '\n')
    print('='*70)
    print('Anomalie')
    print('il numero assoluto di anomalie probabili è ' + str(final_df['true_anomaly'].sum()))
    print('il numero di anomalie, in percentuale, è ' + str(final_df['true_anomaly'].mean() * 100) + '%')
    return final_df

#==================================================================================================================================================================
def create_serie_storiche(filename: str, path: str): 
    df = pd.read_csv(filename, sep = ";") 
    df['unique_id'] = str(1) 
    df['unique_id'] = df['comune_amm'].astype(str) + '_' + df['zona'].astype(str)

    df['linkID'] = 0 
    df.head() 
    date_index = pd.date_range(start= min(df['giorno']), end = max(df['giorno']), freq="H")
    unique_id = df['unique_id'].unique()
        
    # Crea un dizionario di DataFrame: una entry per valore di 'zona'
    dfs = {}
    for omi in unique_id:
        temp = df[df['unique_id'] == omi].copy() 
        temp['ds'] = pd.to_datetime(temp['giorno'].astype(str) + ' ' + temp['ora'].astype(str) + ':00')
        temp = temp.set_index('ds').sort_index() 
        temp = pd.DataFrame(index=date_index).join(temp)
        it_holidays = holidays.IT(years=temp.index.year.unique())
        holiday_dates = set(it_holidays.keys())
        temp['is_holiday'] = [d in holiday_dates for d in temp.index.date]
        temp['unique_id'] = temp['unique_id'].fillna(omi)
        temp['comune_amm'] = temp['comune_amm'].fillna(omi[0:4])
        temp['zona'] = temp['zona'].fillna(omi[5:])
        temp['zona'] = temp['zona'].fillna(omi[0:5])
        temp['velocita_avg'] = temp['velocita_avg'].fillna(0)
        temp['count_traffico'] = temp['count_traffico'].fillna(0)
        temp['day_of_week'] = temp.index.day_name()
        temp['is_weekend'] = temp.index.weekday >= 5
        temp['giorno'] = temp.index.date
        temp['month'] = temp.index.month
        temp['ora'] = temp.index.hour
        temp = temp.reset_index()
        temp = temp.rename(columns= {'index' :'ds'})
        temp['linkID'] = 0
        temp = temp.drop(columns = ['zona', 'comune_amm'])
        dfs[omi] = temp

    os.makedirs(path, exist_ok=True)
    for zona, df_zona in dfs.items():
        print(f'\ncreazione di csv per {zona}')
        out_file = os.path.join(path, f"{zona}_timeseries.csv")
        print(out_file)
        df_zona.to_csv(out_file, index=False)
        # Funzione di ricerca SARIMAX
        
#==================================================================================================================================================================
def format_datetime(df):
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Lunedì, 6=Domenica
    df['day_name'] = df['datetime'].dt.day_name()
    df['day_name'] = df['day_name'].astype('category')
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    df['date'] = df['datetime'].dt.date
    df['is_weekend'] = df['is_weekend'].astype(bool)
    orari = [6, 7, 16, 17]
    df['is_rush_hour'] = df['ora'].isin(orari)
    return df
#==================================================================================================================================================================
def preprocess(df):
    df = df[-365*24-1:-1]
    df['datetime'] = pd.to_datetime(df['ds'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['giorno'] = df['datetime'].dt.date
    df['ora'] = df['datetime'].dt.hour
    df['linkID'] = 0.0
    df['velocita_avg'] = df['velocita_avg'].replace(0, np.nan)
    df['velocita_avg'] = df['velocita_avg'].interpolate(method='linear', limit_direction='both')
    df['velocita_avg'] = df['velocita_avg'].ffill().bfill()
    return df
#==================================================================================================================================================================
def final_preprocess(df, end_test=None):
    """
    df: DataFrame con colonne {"sigla","comune_amm","zona","giorno","ora","count_traffico","velocita_avg"}
    end_test: stringa o datetime. Se fornito, estende la serie fino a questa data.
    """
    df = df.copy()
    all_groups = []
    df.loc[:, 'ds'] = pd.to_datetime(df['giorno'].astype(str) + ' ' + df['ora'].astype(str) + ':00')
    df.loc[:, 'unique_id'] = df['comune_amm'].astype(str) + '_' + df['zona'].astype(str)
    
    grouped = df.groupby('unique_id')
    
    for unique_id, group in grouped:
        # Calcolo del limite temporale
        start_date = group['ds'].min()
        
        # Se end_test non è fornito, usiamo il comportamento originale
        if end_test is None:
            last_timestamp = group['ds'].max()
            current_end_limit = last_timestamp.normalize() - pd.Timedelta(hours=1)
        else:
            current_end_limit = pd.to_datetime(end_test)

        # Generiamo il range esteso
        full_range = pd.date_range(start=start_date, end=current_end_limit, freq='h')
        
        group = group.set_index('ds').sort_index()
        # Il join creerà righe con NaN per le date future (prive di y_real e velocita)
        group = pd.DataFrame(index=full_range).join(group)
        group = group.reset_index().rename(columns={'index': 'datetime'})
        
        # --- RIEMPIMENTO VARIABILI DETERMINISTICHE ---
        # Queste possono essere calcolate anche per il futuro
        group['unique_id'] = unique_id
        group['sigla'] = group['sigla'].bfill().ffill().astype(str)
        group['giorno'] = group['datetime'].dt.date
        group['ora'] = group['datetime'].dt.hour
        
        # Variabili temporali
        group['year'] = group['datetime'].dt.year.astype('uint16')
        group['month'] = group['datetime'].dt.month.astype('uint8')
        group['day_of_week'] = group['datetime'].dt.dayofweek
        group['day_name'] = group['datetime'].dt.day_name().astype('category')
        group['day_of_year'] = group['datetime'].dt.dayofyear
        group['week_of_year'] = group['datetime'].dt.isocalendar().week
        group['is_weekend'] = group['datetime'].dt.weekday >= 5
        
        # Holiday e DST (funzionano nel futuro)
        it_holidays = holidays.IT(years=group['datetime'].dt.year.unique())
        holiday_dates = set(it_holidays.keys())
        group['is_holiday'] = [d in holiday_dates for d in group['datetime'].dt.date]
        
        group = apply_dst_correction(group)
        group['is_rush_hour'] = group['hour_corrected'].isin([6, 7, 16, 17])
        
        # --- GESTIONE VALORI MANCANTI (X vs Y) ---
        group['linkID'] = 0.0
        # Per il futuro, count_traffico e velocita_avg rimarranno NaN o 0. 
        # Nota: y_real (count_traffico) DEVE essere NaN nel futuro per non inquinare i test.
        group['count_traffico'] = group['count_traffico'].fillna(0) 

        all_groups.append(group)
    
    final_df = pd.concat(all_groups, ignore_index=True)
    # Riordino finale colonne
    cols = ['datetime','sigla','unique_id','month','giorno','day_of_week',
            'is_holiday','ora','hour_sensor','hour_corrected','is_rush_hour',
            'is_dst','linkID','count_traffico']
    return final_df[cols]
#==================================================================================================================================================================
def run_rolling_window_evaluation(df, y, X, horizon, periods, TRAIN_WINDOW = 365*24, model_type='OLS'):
    """
    Versione Rolling Window con Offset.
    - horizon: quante ore prevedere (es. 48)
    - periods: offset in GIORNI rispetto alla fine dei dati (0 = ultimi dati disponibili)
    """
    # Parametri finestra
    test_size = horizon
    # Definiamo un training size fisso (es. 1 anno o tutto il disponibile fino al test)
    # Per coerenza, usiamo tutto quello che precede il test window
    
    # Calcolo dell'offset: lo spostamento all'indietro è p * 24 ore
    offset = periods * 24
    
    # Indici di fine e inizio
    end_idx = len(df) - offset
    start_test_idx = end_idx - test_size
    
    # Se l'offset è troppo grande per i dati disponibili
    if start_test_idx < 0:
        raise ValueError(f"Offset {periods} giorni troppo grande per la lunghezza dei dati.")

    # Split dei dati basato sulla finestra mobile
    y_train = y.iloc[start_test_idx-TRAIN_WINDOW:start_test_idx]
    y_test  = y.iloc[start_test_idx:end_idx]
    
    X_train = X.iloc[start_test_idx-TRAIN_WINDOW:start_test_idx, :]
    X_test  = X.iloc[start_test_idx:end_idx, :]
    
    ds_test = df['datetime'].iloc[start_test_idx:end_idx]

    # Allineamento e aggiunta costante
    X_train_const, X_test_const = ensure_const_and_align(X_train, X_test)

    # --- LOGICA DISCRIMINANTE ---
    m_type = model_type.upper()
    
    if m_type == 'OLS':
        return _run_ols_logic(y_train, y_test, X_train_const, X_test_const, ds_test)
        
    elif m_type == 'NEGBIN':
        return _run_negbin_logic(y_train, y_test, X_train_const, X_test_const, ds_test)
    
    elif m_type == 'NEGBIN+SARIMAX':
        return _run_negbin_sarimax_logic(y_train, y_test, X_train_const, X_test_const, X, ds_test, df)
    else:
        raise ValueError(f"Modello {model_type} non supportato.")
        # Calcolo metriche comuni per il print (scala lineare)
#==================================================================================================================================================================

def detect_consistent_anomalies(results_df, min_daily_weight=3.0):
    # 1. Identificazione puntuale dell'anomalia probabile
    results_df['probable_anomaly'] = (results_df['y_real'] < results_df['lower_99']) | (results_df['y_real'] > results_df['upper_99'])
    
    # Assicuriamoci che 'ds' sia datetime
    results_df['ds'] = pd.to_datetime(results_df['ds'])
    
    # 2. Definizione dei pesi in base all'ora
    # Estraiamo l'ora
    hours = results_df['ds'].dt.hour
    
    # Creiamo una serie di pesi: 0.5 per la fascia 22-03, 1.0 per il resto
    # La fascia 22-03 include: 22, 23, 0, 1, 2, 3
    night_mask = (hours >= 22) | (hours <= 3)
    results_df['anomaly_weight'] = 1.0
    results_df.loc[night_mask, 'anomaly_weight'] = 0.5
    
    # 3. Calcolo del punteggio (weight * bool)
    # Solo le righe che sono 'probable_anomaly' contribuiscono al punteggio
    results_df['weighted_score'] = results_df['probable_anomaly'] * results_df['anomaly_weight']
    
    # 4. Aggregazione giornaliera del punteggio
    results_df['date_only'] = results_df['ds'].dt.date
    daily_total_weight = results_df.groupby('date_only')['weighted_score'].transform('sum')
    
    # 5. Definizione is_anomaly finale
    # L'anomalia è confermata se il punto è fuori dalle bande E il peso totale del giorno supera la soglia
    results_df['is_anomaly'] = (results_df['probable_anomaly']) & (daily_total_weight >= min_daily_weight)
    
    # Pulizia colonne di servizio
    results_df.drop(columns=['date_only', 'anomaly_weight', 'weighted_score'], inplace=True)
    
    return results_df

#==================================================================================================================================================================
def create_from_models( X_train,
                        y_train,
                        model_type = 'OLS'
                        ):
    """
    Funzione universale per la valutazione rolling.
    Modelli supportati: 'OLS' (OLS Log + SARIMAX), 'NegBin', 'Prophet'
    """
    # Aggiungo costante
    X_train_const = sm.add_constant(X_train, has_constant = 'add')
    # --- LOGICA DISCRIMINANTE ---
    if model_type.upper() == 'OLS':
        models = create_ols_model(y_train, X_train_const)
    elif model_type.upper() == 'NEGBIN':
        models = create_negbin_model(y_train, X_train_const)
    elif model_type.upper() == 'NEGBIN+SARIMAX':
        models = create_negbin_sarimax_model(y_train, X_train_const)
    elif model_type.upper() == 'GAUSSIAN':
        models = create_gaussian_glm_model(y_train, X_train_const)
    else:
        raise ValueError(f"Modello {model_type} non supportato.")

    # Calcolo metriche comuni per il print (scala lineare)
    
    return model_type, models
#==================================================================================================================================================================
import matplotlib.pyplot as plt

def plot_model_results(results_df, model_name, provincia=None, comune=None, zona_omi=None):    
    """
    Renders an advanced plot with automatic anomaly highlighting.
    """
    # 2. Immagine più grande in verticale (altezza raddoppiata: 14)
    plt.figure(figsize=(45, 14), dpi=100) 
    
    # 4. Traduzione in inglese e formattazione titolo
    info_territorio = f" - Province: {provincia.upper()} | Municipality: {comune.upper()} | OMI Zone: {zona_omi.upper()}" if comune else ""
    title_text = f"Anomaly Detection & Forecast | Model: {model_name.upper()}{info_territorio}"
    
    #1. PLOT DEI DATI BASE ---
    # Valore Reale (3. Linea più scura aumentando l'alpha da 0.4 a 0.75)
    plt.plot(results_df.index, results_df["y_real"], 
             label="Actual Value", color="#2C3E50", alpha=0.75, linewidth=4, zorder=1)
    
    # Predizione Point Forecast (Tradotto)
    plt.plot(results_df.index, results_df["y_pred"], 
             label=f"Prediction ({model_name})", color="#E67E22", linewidth=5, zorder=2)
    
    #2. INTERVALLO DI CONFIDENZA
    lower_col = "lower_99" if "lower_99" in results_df.columns else "y_lower"
    upper_col = "upper_99" if "upper_99" in results_df.columns else "y_upper"
    
    if lower_col in results_df.columns:
        plt.fill_between(
            results_df.index,
            results_df[lower_col],
            results_df[upper_col],
            alpha=0.25,
            label="Confidence Interval (99%)", # Tradotto
            color="#3498DB",
            zorder=0
        )

    #3. EVIDENZIAZIONE ANOMALIE
    if "is_anomaly" in results_df.columns:
        anomalies = results_df[results_df["is_anomaly"] == True]
        
        if not anomalies.empty:
            plt.scatter(
                anomalies.index, 
                anomalies["y_real"], 
                color="#52aeff",
                label="Detected Anomalies", # Tradotto
                edgecolors="black",   
                s=200,                 
                marker="o",           
                zorder=3              
            )

    #4. ESTETICA
    # 2. Titolo centrato (loc='center') e ingrandito
    plt.title(title_text, fontsize=30, fontweight='bold', pad=25, loc='center')
    
    # 3. Aumentato font didascalie (da 12 a 16) e tradotte
    plt.xlabel("Date", fontsize=25)
    plt.ylabel("Value", fontsize=25)
    
    # 1. Dimensione tick su assi X e Y
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    
    # 3. Aumentato font legenda (22)
    plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=30)
    
    # Pulizia bordi
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()