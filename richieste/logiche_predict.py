import pandas as pd
import numpy as np
import statsmodels.api as sm
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsforecast import StatsForecast
from statsforecast.models import MSTL
from scipy.stats import nbinom
from arch import arch_model
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import Sequential, layers, models, backend as K
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from datetime import timedelta



def apply_ewma_volatility(eps_t, lambda_param=0.94, horizon=48):
    """
    Calcola la volatilità EWMA invece del GARCH.
    lambda_param: 0.94 è lo standard RiskMetrics per dati ad alta frequenza.
    """
    # 1. Inizializzazione della varianza (usiamo la varianza dei residui iniziali)
    n = len(eps_t)
    vols = np.zeros(n)
    vols[0] = eps_t.var() 

    # 2. Calcolo ricorsivo EWMA (In-sample)
    for i in range(1, n):
        vols[i] = (1 - lambda_param) * (eps_t.iloc[i-1]**2) + lambda_param * vols[i-1]
    
    # La volatilità è la radice quadrata della varianza
    sigma_series = np.sqrt(vols)
    
    # 3. Forecast per l'horizon (nel EWMA il forecast della varianza è costante)
    # sigma_{t+h} = sigma_{t+1}
    last_vol = vols[-1]
    forecast_vol = np.sqrt((1 - lambda_param) * (eps_t.iloc[-1]**2) + lambda_param * last_vol)
    
    # Creiamo un array di previsioni costante per l'horizon richiesto
    sigma_hat_fc = np.full(horizon, forecast_vol)
    
    return sigma_series, sigma_hat_fc
#===================================================================================================================================================
def apply_sarimax(df):
    """
    Versione ultra-veloce: elimina la Grid Search e usa un ordine prefissato robusto.
    """
    resid = df["residuals"].dropna()

    # Invece di provare 16 combinazioni, usiamo un ordine standard (1,0,1) 
    # che cattura la maggior parte della correlazione residua senza esplodere computazionalmente.
    fixed_order = (1, 0, 1)
    fixed_seasonal = (1, 0, 0, 24) # Solo AR stagionale per stabilità

    try:
        model = SARIMAX(
            resid,
            order=fixed_order,
            seasonal_order=fixed_seasonal,
            trend="n",
            enforce_stationarity=False, # Velocizza la convergenza
            enforce_invertibility=False # Riduce i vincoli sul solver
        )
        # maxiter ridotto e metodo più rapido per una convergenza veloce
        res = model.fit(disp=False, maxiter=20, method='nm') 
        best_model = res
        best_order = fixed_order
        best_seasonal = fixed_seasonal
    except Exception:
        # Fallback estremo: se anche il SARIMA fallisce, restituiamo un modello nullo (identità)
        return None, None, None

    return best_order, best_seasonal, best_model
#===================================================================================================================================================

def _run_ols_sarimax_logic(y_train, y_test, X_train, X_test, ds_test, test_size):
    """
    Logica Ibrida Velocizzata.
    """
    y_train_log = np.log1p(y_train)
    ols_model = sm.OLS(y_train_log, X_train).fit()
    
    resid_ols = y_train_log - ols_model.predict(X_train)
    
    # Chiamata alla nuova apply_sarimax snella
    _, _, sarima_fit = apply_sarimax(pd.DataFrame({'residuals': resid_ols}))
    
    if sarima_fit is not None:
        mu_resid_hat = sarima_fit.get_forecast(steps=test_size).predicted_mean.values
        sarima_resid = sarima_fit.resid
    else:
        # Se il SARIMA fallisce, usiamo solo l'OLS (zero residui previsti)
        mu_resid_hat = np.zeros(test_size)
        sarima_resid = resid_ols

    ols_forecast_log = ols_model.predict(X_test).values
    y_pred_log = ols_forecast_log + mu_resid_hat
    
    # 2. Fit GARCH velocizzato (opzionale: se è ancora lento, si può semplificare)
    # Usiamo un modello GARCH(1,1) standard che solitamente è rapido
    try:
        garch_model = arch_model(sarima_resid, p=1, q=1, vol='Garch', dist='normal', rescale=False)
        res_garch = garch_model.fit(disp='off', show_warning=False)
        garch_forecast = res_garch.forecast(horizon=test_size, reindex=False)
        sigma_garch = np.sqrt(garch_forecast.variance.values[-1, :])
    except:
        # Fallback deviazione standard storica se GARCH non converge
        sigma_garch = np.repeat(np.std(sarima_resid), test_size)

    # 3. Formattazione
    res = pd.DataFrame({
        'ds': ds_test.values,
        'y_real': y_test.values,
        'y_pred_log': y_pred_log,
        'y_pred': np.expm1(y_pred_log),
        'sigma': sigma_garch
    })

    res['lower_99'] = np.expm1(res['y_pred_log'] - 2.576 * res['sigma']).clip(lower=0)
    res['upper_99'] = np.expm1(res['y_pred_log'] + 2.576 * res['sigma'])

    return res[['ds', 'y_real', 'y_pred', 'lower_99', 'upper_99']]
#===================================================================================================================================================
def _run_ols_logic(y_train, y_test, X_train, X_test, ds_test):
    # 1. Fit nel dominio log
    y_train_log = np.log1p(y_train)
    model = sm.OLS(y_train_log, X_train).fit()
    
    y_pred_log = model.predict(X_test)
    train_preds_log = model.predict(X_train)
    
    resid = y_train_log - train_preds_log
    train_dates = pd.date_range(start="2024-01-01 00:00:00", periods=len(y_train), freq='H')
    
    calib_df = pd.DataFrame({
        'resid': resid,
        'month': train_dates.month,
        'hour': train_dates.hour
    })
    global_std = np.std(resid)
    # reset_index è fondamentale per il merge successivo
    hourly_monthly_std = calib_df.groupby(['month', 'hour'])['resid'].std().reset_index()
    
    res = pd.DataFrame({
        'ds': ds_test.values,
        'y_real': y_test.values,
        'y_pred': np.expm1(y_pred_log),
        'y_pred_log': y_pred_log,
        'month': ds_test.dt.month.values,
        'hour': ds_test.dt.hour.values
    })
    
    res = res.merge(hourly_monthly_std, on=['month', 'hour'], how='left')
    
    res['resid'] = res['resid'].fillna(global_std)
    
    #CALCOLO BANDE AL 99% (Z=2.576)
    res['lower_99'] = np.expm1(res['y_pred_log'] - (2.576 * res['resid'])).clip(lower=0)
    res['upper_99'] = np.expm1(res['y_pred_log'] + (2.576 * res['resid']))

    return res[['ds', 'y_real', 'y_pred', 'lower_99', 'upper_99']]
#===================================================================================================================================================
def _run_ols_sarimax_logic(y_train, y_test, X_train, X_test, ds_test, test_size):
    """
    Logica Ibrida: OLS (Base) + SARIMAX (Residui Media) + GARCH (Volatilità Bande).
    """
    # 1. STEP MEDIA: OLS + SARIMAX
    y_train_log = np.log1p(y_train)
    ols_model = sm.OLS(y_train_log, X_train).fit()
    
    # Residui OLS da passare al SARIMAX
    resid_ols = y_train_log - ols_model.predict(X_train)
    _, _, sarima_fit = apply_sarimax(pd.DataFrame({'residuals': resid_ols}))
    
    # Previsione puntuale della componente stocastica
    mu_resid_hat = sarima_fit.get_forecast(steps=test_size).predicted_mean.values
    ols_forecast_log = ols_model.predict(X_test).values
    y_pred_log = ols_forecast_log + mu_resid_hat
    sarima_resid = sarima_fit.resid
    
    # Fit GARCH
    garch_model = arch_model(sarima_resid, p=1, q=1, vol='Garch', dist='normal', rescale=False)
    res_garch = garch_model.fit(disp='off')
    
    # Previsione volatilità per l'orizzonte di test
    garch_forecast = res_garch.forecast(horizon=test_size, reindex=False)
    sigma_garch = np.sqrt(garch_forecast.variance.values[-1, :])

    # 3. FORMATTAZIONE E BANDE
    res = pd.DataFrame({
        'ds': ds_test.values,
        'y_real': y_test.values,
        'y_pred_log': y_pred_log,
        'y_pred': np.expm1(y_pred_log),
        'sigma': sigma_garch
    })

    res['lower_99'] = np.expm1(res['y_pred_log'] - 2.576 * res['sigma']).clip(lower=0)
    res['upper_99'] = np.expm1(res['y_pred_log'] + 2.576 * res['sigma'])

    return res[['ds', 'y_real', 'y_pred', 'lower_99', 'upper_99']]

#===================================================================================================================================================

def _run_autoets_logic(y_train, y_test, X_train, X_test, ds_test):
    """
    Modello OLS per la media e GARCH(1,1) per la volatilità delle bande.
    Risolve il problema della divergenza delle bande su lunghi periodi.
    """
    y_train_log = np.log1p(y_train)
    ols_model = sm.OLS(y_train_log, X_train).fit()
    
    resid_train = ols_model.predict(X_train) - y_train_log
    
    # 2. FIT GARCH(1,1) sui residui
    garch_model = arch_model(resid_train, p=1, q=1, vol='Garch', dist='normal', rescale=False)
    res_garch = garch_model.fit(disp='off')
    
    # 3. PREVISIONE (15 giorni = test_size passi)
    horizon = len(X_test)
    
    # Predizione della media (OLS)
    y_pred_log = ols_model.predict(X_test).values
    
    # Previsione della Volatilità (GARCH)
    # forecast.variance è la varianza attesa per ogni step futuro
    garch_forecast = res_garch.forecast(horizon=horizon, reindex=False)
    vols_forecast = np.sqrt(garch_forecast.variance.values[-1, :]) 
    
    # 4. COSTRUZIONE OUTPUT
    res = pd.DataFrame({
        'ds': ds_test.values,
        'y_real': y_test.values,
        'y_pred_log': y_pred_log,
        'y_pred': np.expm1(y_pred_log),
        'sigma_garch': vols_forecast
    })
    
    # 5. BANDE AL 99% (Z = 2.576)
    # Il GARCH stabilizza sigma_garch rendendo le bande sode e non divergenti
    res['lower_99'] = np.expm1(res['y_pred_log'] - (2.576 * res['sigma_garch'])).clip(lower=0)
    res['upper_99'] = np.expm1(res['y_pred_log'] + (2.576 * res['sigma_garch']))
    
    return res[['ds', 'y_real', 'y_pred', 'lower_99', 'upper_99']]
#===================================================================================================================================================

def _run_negbin_sarimax_logic(y_train, y_test, X_train, X_test, X_global, ds_test, df):
    if 'const' in X_train.columns and 'const' not in X_global.columns:
        X_global = sm.add_constant(X_global)

    start_test = pd.to_datetime(ds_test.iloc[0])
    start_warmup = start_test - timedelta(days=7)

    model_nb = sm.NegativeBinomial(y_train, X_train).fit(disp=False)
    mu_base = model_nb.predict(X_test).values
    alpha_hat = np.clip(model_nb.params['alpha'], 1e-6, 1e6)

    train_resid = y_train - model_nb.predict(X_train)
    sarima_res = sm.tsa.SARIMAX(
        train_resid,
        order=(0, 0, 0),
        seasonal_order=(1, 0, 0, 24),
        enforce_stationarity=False
    ).fit(disp=False)

    warmup_mask = (df['datetime'] >= start_warmup) & (df['datetime'] < start_test)
    df_warmup = df.loc[warmup_mask]
    X_warmup = X_global.loc[df_warmup.index, X_train.columns]
    y_warmup = df_warmup['count_traffico'].values

    warmup_preds = model_nb.predict(X_warmup)
    warmup_resid = y_warmup - warmup_preds
    warmup_resid.index = pd.RangeIndex(start=sarima_res.nobs, stop=sarima_res.nobs + len(warmup_resid))
    updated_sarima = sarima_res.extend(warmup_resid)

    # Previsione E volatilità (Standard Error) del SARIMAX, non solo la media
    sarima_forecast_obj = updated_sarima.get_forecast(steps=len(y_test))
    res_forecast = sarima_forecast_obj.predicted_mean
    sarima_stderr = sarima_forecast_obj.se_mean

    y_pred_final = np.maximum(mu_base + res_forecast, 0)

    # Alpha dinamico: infla la dispersione in proporzione all'incertezza del forecast SARIMAX,
    # coerentemente con ModelPredictor.predict()
    volatility_scaling = 1 + (sarima_stderr / (y_pred_final + 1e-6))
    alpha_dynamic = alpha_hat * volatility_scaling

    n_param = 1.0 / alpha_dynamic
    p_param = n_param / (n_param + np.maximum(y_pred_final, 1e-6))

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        lower_99 = nbinom.ppf(0.01, n_param, p_param)
        upper_99 = nbinom.ppf(0.99, n_param, p_param)

        lower_99 = np.floor(np.nan_to_num(np.maximum(0, lower_99), nan=0.0))
        upper_99 = np.ceil(np.nan_to_num(upper_99, nan=np.max(y_pred_final) * 2))

    return pd.DataFrame({
        'ds': ds_test.values,
        'y_real': y_test.values,
        'y_pred': y_pred_final,
        'lower_99': lower_99,
        'upper_99': upper_99,
        'model_alpha': alpha_hat
    })
#===================================================================================================================================================
def _run_negbin_logic(y_train, y_test, X_train, X_test, ds_test):
    """
    Logica Negative Binomial corretta: 
    Usa sm.NegativeBinomial per stimare correttamente l'alpha (overdispersion).
    """
    model_nb = sm.NegativeBinomial(y_train, X_train).fit(disp=False)
    mu_hat = model_nb.predict(X_test).values
    alpha_hat = model_nb.params['alpha']
    
    n = 1.0 / alpha_hat
    p = n / (n + mu_hat)
    
    lower_99 = nbinom.ppf(0.01, n, p)
    upper_99 = nbinom.ppf(0.99, n, p)
    
    lower_99 = np.nan_to_num(lower_99, nan=0.0)
    upper_99 = np.nan_to_num(upper_99, nan=np.max(mu_hat)*2)

    #COSTRUZIONE OUTPUT
    return pd.DataFrame({
        'ds': ds_test.values,
        'y_real': y_test.values,
        'y_pred': mu_hat,
        'lower_99': lower_99,
        'upper_99': upper_99,
    })
#===================================================================================================================================================

def _run_mstl_logic(y_train, y_test, X_train, X_test, ds_test):
    # 1. Preparazione DF (uguale a prima)
    train_dates = pd.date_range(start="2024-01-01", periods=len(y_train), freq='H')
    df_train = pd.DataFrame({
        'unique_id': 1,
        'ds': train_dates,
        'y': np.log1p(y_train.values)
    })

    # 2. Fit del Modello
    # Se vuoi bande che tengano conto dell'incertezza storica, 
    # StatsForecast le calcola durante il predict se specifichi 'level'
    models = [MSTL(season_length=[24, 168])]
    sf = StatsForecast(models=models, freq='H', n_jobs=-1)
    sf.fit(df_train)

    # 3. Previsione con Intervalli Probabilistici
    # Il 'level=[99]' attiva il calcolo degli intervalli al 99%
    # StatsForecast userà la distribuzione dei residui del fit per calcolarli
    forecast_df = sf.predict(h=len(y_test), level=[99])
    
    # 4. Formattazione
    res = pd.DataFrame({
        'ds': ds_test.values,
        'y_real': y_test.values,
        'y_pred_log': forecast_df['MSTL'].values,
        'y_pred': np.expm1(forecast_df['MSTL'].values),
        'lower_99_log': forecast_df['MSTL-lo-99'].values,
        'upper_99_log': forecast_df['MSTL-hi-99'].values
    })

    # Portiamo fuori dal log
    res['lower_99'] = np.expm1(res['lower_99_log']).clip(lower=0)
    res['upper_99'] = np.expm1(res['upper_99_log'])

    return res[['ds', 'y_real', 'y_pred', 'lower_99', 'upper_99']]

#=======================================================================================================================================

def _run_mlp_logic(y_train, y_test, X_train, X_test, ds_test, test_size):
    """
    Versione Evoluta: MLP con Standardizzazione e 
    Bande di Confidenza Dinamiche basate sull'errore orario residuo.
    """
    # 1. Normalizzazione
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    # 2. Definizione del modello (Manteniamo i tuoi ottimi parametri)
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        learning_rate='adaptive',
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )

    # 3. Fit
    model.fit(X_train_scaled, y_train_scaled)

    # 4. Previsione
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # 5. CALCOLO BANDE DINAMICHE (L'upgrade)
    # Calcoliamo i residui sul training (valori reali - valori predetti dalla rete)
    y_train_pred_scaled = model.predict(X_train_scaled)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    residuals = y_train.values - y_train_pred

    # Creiamo una mappa dell'errore basata sull'ora (usando l'indice del training)
    # Assumiamo che X_train conservi l'ordine orario
    train_hours = pd.date_range(start="2024-01-01", periods=len(y_train), freq='H').hour
    
    error_map = pd.DataFrame({
        'res': residuals,
        'hour': train_hours
    }).groupby('hour')['res'].std().reset_index()
    
    # Global std come fallback
    global_std = np.std(residuals)

    # 6. Formattazione Output
    res = pd.DataFrame({
        'ds': ds_test.values,
        'y_real': y_test.values,
        'y_pred': np.maximum(0, y_pred),
        'hour': ds_test.dt.hour.values
    })

    # Integriamo la deviazione standard specifica per ogni ora
    res = res.merge(error_map, on='hour', how='left').fillna(global_std)

    # Calcolo bande al 99% (Z=2.576)
    # Ora la banda si allarga e si stringe seguendo il 'ritmo' dell'incertezza reale
    res['lower_99'] = (res['y_pred'] - (2.576 * res['res'])).clip(lower=0)
    res['upper_99'] = res['y_pred'] + (2.576 * res['res'])

    return res[['ds', 'y_real', 'y_pred', 'lower_99', 'upper_99']]

#===================================================================================================================================================

def _run_keras_lstm_logic(y_train, y_test, X_train, X_test, ds_test, test_size):
    """
    Versione corretta per evitare errori NoneType e gestire i dati in input.
    """
    
    # --- FIX 1: Controllo robustezza input ---
    # Se i dati sono None o vuoti, restituiamo un DataFrame vuoto o gestiamo l'errore
    if y_train is None or X_train is None or ds_test is None:
        print("Errore: Uno dei parametri di input è None.")
        return pd.DataFrame()

    # --- FIX 2: Gestione dei valori mancanti (NaN) ---
    # I modelli Keras non accettano NaN
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    y_train = y_train.fillna(0)

    # 1. Normalizzazione
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    
    # Assicuriamoci che y_train sia un array 2D per lo scaler
    y_train_values = y_train.values.reshape(-1, 1)
    y_train_scaled = scaler_y.fit_transform(y_train_values)

    # 2. Reshaping per LSTM [samples, time_steps, features]
    # Usiamo .shape per assicurarci che gli array siano validi
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # 3. Architettura della Rete
    model = Sequential([
        Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
        LSTM(50, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # 4. Training
    model.fit(
        X_train_reshaped,
        y_train_scaled,
        epochs=30,
        batch_size=16,
        verbose=0
    )

    # 5. Previsione
    y_pred_scaled = model.predict(X_test_reshaped, verbose=0)

    # Invertiamo la scala
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    # 6. Calcolo dell'incertezza
    y_train_pred_scaled = model.predict(X_train_reshaped, verbose=0)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()
    
    # --- FIX 3: Allineamento shape per i residui ---
    # Usiamo .values per evitare conflitti di indice se y_train è una Series
    residui = y_train.values.flatten() - y_train_pred
    std_error = np.std(residui)

    # --- FIX 4: Protezione contro lunghezze diverse ---
    # Costruiamo il dizionario assicurandoci che tutti gli array abbiano la stessa lunghezza
    res_df = pd.DataFrame({
        'ds': ds_test.values if hasattr(ds_test, 'values') else ds_test,
        'y_real': y_test.values if hasattr(y_test, 'values') else y_test,
        'y_pred': np.maximum(0, y_pred),
        'lower_99': np.maximum(0, y_pred - 2.58 * std_error),
        'upper_99': y_pred + 2.58 * std_error
    })
    return res_df

#=======================================================================================================================================
def _run_prophet_logic(train_prophet, y_test, X_test_const, ds_test, test_size):
    """
    Logica interna per Prophet con regressori esogeni e clipping a zero di tutte le stime.
    """
    # 1. Inizializzazione (99% CI richiesto dal tuo plot -> interval_width=0.99)
    model = Prophet(interval_width=0.99, daily_seasonality=False, 
                    weekly_seasonality=False, yearly_seasonality=False)

    # 2. Aggiunta regressori
    exogenous_cols = [col for col in train_prophet.columns if col not in ['ds', 'y', 'const']]
    for col in exogenous_cols:
        model.add_regressor(col)

    # 3. Fit
    model.fit(train_prophet)

    # 4. Future Dataframe
    future = pd.DataFrame({'ds': ds_test.values})
    for col in exogenous_cols:
        future[col] = X_test_const[col].values

    # 5. Prediction
    forecast = model.predict(future)
    
    # 6. Risultati con CLIPPING a 0 per tutte le bande
    results_df = pd.DataFrame({
        'ds': ds_test.values,
        'y_real': y_test.values,
        'y_pred': forecast['yhat'].clip(lower=0).values,
        'lower_99': forecast['yhat_lower'].clip(lower=0).values,
        'upper_99': forecast['yhat_upper'].clip(lower=0).values
    })
    
    return results_df
#=======================================================================================================================================

def pinball_loss(q, y_true, y_pred):
    """Funzione di perdita per la regressione quantilica usando TensorFlow."""
    err = y_true - y_pred
    # Usiamo tf.reduce_mean invece di K.mean e tf.maximum invece di K.maximum
    return tf.reduce_mean(tf.maximum(q * err, (q - 1) * err), axis=-1)

def _run_keras_quantile_logic(y_train, y_test, X_train, X_test, ds_test):
    # 1. Pre-processing
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    # 2. Architettura del Modello
    inputs = layers.Input(shape=(X_train.shape[1],))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Tre output per i quantili: 0.5% (Lower), 50% (Mediana), 99.5% (Upper)
    out_low = layers.Dense(1, name='low')(x)
    out_mid = layers.Dense(1, name='mid')(x)
    out_high = layers.Dense(1, name='high')(x)
    
    model = models.Model(inputs=inputs, outputs=[out_low, out_mid, out_high])

    # 3. Compilazione con Pinball Loss differenziata
    model.compile(
        optimizer='adam',
        loss={
            'low': lambda y_true, y_pred: pinball_loss(0.005, y_true, y_pred),
            'mid': lambda y_true, y_pred: pinball_loss(0.500, y_true, y_pred),
            'high': lambda y_true, y_pred: pinball_loss(0.995, y_true, y_pred)
        }
    )

    # 4. Training
    model.fit(
        X_train_scaled, 
        {'low': y_train_scaled, 'mid': y_train_scaled, 'high': y_train_scaled},
        epochs=100, batch_size=32, verbose=0, validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    )

    # 5. Prediction e Inverse Scaling
    preds_scaled = model.predict(X_test_scaled)
    
    lower_99 = scaler_y.inverse_transform(preds_scaled[0]).flatten()
    y_pred = scaler_y.inverse_transform(preds_scaled[1]).flatten()
    upper_99 = scaler_y.inverse_transform(preds_scaled[2]).flatten()

    # 6. Output Standard
    return pd.DataFrame({
    'ds': ds_test.values,
    'y_real': y_test.values,
    'y_pred': np.maximum(0, y_pred),
    'lower_99': np.maximum(0, lower_99), # Clip a zero per dati non negativi
    'upper_99': np.maximum(0, upper_99)
    })