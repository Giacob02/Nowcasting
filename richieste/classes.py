import numpy as np
import pandas as pd
from datetime import timedelta
import statsmodels.api as sm
import warnings
from scipy.stats import nbinom
ALPHA = 0.01
LCL = ALPHA/2.0
UCL = 1 - (ALPHA / 2.0)

class ModelPredictorNB:
    def __init__(self, model_nb, alpha):
        self.model_nb = model_nb
        self.alpha = alpha
#============================================================================================================================
    def predict(self, X_test, y_test, X_global, ds_test, df):
            if 'const' in X_test.columns and 'const' not in X_global.columns:
                X_global = sm.add_constant(X_global)

            model_nb = self.model_nb
            mu_base = model_nb.predict(X_test).values
            alpha_hat = np.clip(model_nb.params['alpha'], 1e-6, 1e6)

            # Nessuna correzione SARIMAX: la previsione finale è direttamente mu_base
            y_pred_final = mu_base

            n_param = 1.0 / alpha_hat
            p_param = n_param / (n_param + np.maximum(y_pred_final, 1e-6))

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                lower_99 = nbinom.ppf(LCL, n_param, p_param)
                upper_99 = nbinom.ppf(UCL, n_param, p_param)

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
    
class ModelPredictor:    
    def __init__(self, model_nb, model_sarima, alpha):
        self.model_nb = model_nb
        self.model_sarima = model_sarima
        self.alpha = alpha
#============================================================================================================================
    def predict(self, X_test, y_test, X_global, ds_test, df):   
            if 'const' in X_test.columns and 'const' not in X_global.columns:
                X_global = sm.add_constant(X_global)

            start_test = pd.to_datetime(ds_test.iloc[0])
            start_warmup = start_test - timedelta(days=7)
            model_nb = self.model_nb
            mu_base = model_nb.predict(X_test).values
            alpha_hat = np.clip(model_nb.params['alpha'], 1e-6, 1e6)

            sarima_res = self.model_sarima
            
            warmup_mask = (df['datetime'] >= start_warmup) & (df['datetime'] < start_test)
            df_warmup = df.loc[warmup_mask]
            X_warmup = X_global.loc[df_warmup.index, X_global.columns]
            y_warmup = df_warmup['count_traffico'].values
            
            warmup_preds = model_nb.predict(X_warmup)
            warmup_resid = y_warmup - warmup_preds
            warmup_resid.index = pd.RangeIndex(start=sarima_res.nobs, stop=sarima_res.nobs + len(warmup_resid))
            updated_sarima = sarima_res.extend(warmup_resid)
            
            # Previsione e volatilità (Standard Error) del SARIMAX
            sarima_forecast_obj = updated_sarima.get_forecast(steps=len(y_test))
            res_forecast = sarima_forecast_obj.predicted_mean
            sarima_stderr = sarima_forecast_obj.se_mean
            
            # y_pred_final = np.maximum(mu_base + res_forecast, 0)
            y_pred_final = mu_base + res_forecast
            volatility_scaling = 1 + (sarima_stderr / (y_pred_final + 1e-6))
            alpha_dynamic = alpha_hat * volatility_scaling
            
            n_param = 1.0 / alpha_dynamic
            p_param = n_param / (n_param + np.maximum(y_pred_final, 1e-6))

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                # Calcolo quantili direttamente sulla distribuzione "corretta" dalla volatilità
                lower_99 = nbinom.ppf(LCL, n_param, p_param)
                upper_99 = nbinom.ppf(UCL, n_param, p_param)
                
                # Gestione pulita dello zero e arrotondamento all'unità
                # np.floor assicura che 0.001 diventi 0, includendo il valore nel bound
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

#============================================================================================================================
    def predict_velo(self, X_test, y_test, X_global, ds_test, df):
        # 1. Gestione Costante (sm.add_constant)
        if 'const' in X_test.columns and 'const' not in X_global.columns:
            X_global = sm.add_constant(X_global)

        start_test = pd.to_datetime(ds_test.iloc[0])
        start_warmup = start_test - timedelta(days=7)
        
        model_nb = self.model_nb
        sarima_res = self.model_sarima
        alpha_hat = np.clip(model_nb.params.get('alpha', 0.1), 1e-6, 1e6)

        # 2. Estrazione Warmup con controllo di esistenza
        warmup_mask = (df['datetime'] >= start_warmup) & (df['datetime'] < start_test)
        df_warmup = df.loc[warmup_mask]
        
        if df_warmup.empty:
            # Se non ho warmup, uso il SARIMAX base senza estensione
            updated_sarima = sarima_res
            mu_base = model_nb.predict(X_test).values
        else:
            X_warmup = X_global.loc[df_warmup.index, X_global.columns]
            y_warmup = df_warmup['count_traffico'].values # Assicurati che sia la target corretta
            
            warmup_preds = model_nb.predict(X_warmup)
            warmup_resid = y_warmup - warmup_preds
            
            # --- MIGLIORAMENTO: Clipping dei residui ---
            # Evita che un'anomalia nell'ultima ora del warmup spari la previsione alle stelle
            std_resid_train = np.std(sarima_res.resid)
            warmup_resid = np.clip(warmup_resid, -3 * std_resid_train, 3 * std_resid_train)
            
            # --- MIGLIORAMENTO: Allineamento Indici ---
            warmup_resid.index = pd.RangeIndex(start=sarima_res.nobs, stop=sarima_res.nobs + len(warmup_resid))
            updated_sarima = sarima_res.extend(warmup_resid)
            
            mu_base = model_nb.predict(X_test).values

        # 3. Previsione SARIMAX
        sarima_forecast_obj = updated_sarima.get_forecast(steps=len(y_test))
        res_forecast = sarima_forecast_obj.predicted_mean
        sarima_stderr = sarima_forecast_obj.se_mean
        
        # --- CORREZIONE: Stabilizzazione del Forecast ---
        # Se noti ancora sbalzi, puoi applicare un 'damping' (smorzamento) ai primi step
        # res_forecast[:3] = res_forecast[:3] * 0.5 

        y_pred_final = mu_base + res_forecast
        
        # 4. Dinamica della Volatilità e Distribuzione NegBin
        # Assicuriamoci che y_pred_final non sia negativo o troppo vicino a zero
        y_pred_final = np.maximum(y_pred_final, 1e-6)
        
        volatility_scaling = 1 + (sarima_stderr / (y_pred_final + 1e-6))
        alpha_dynamic = alpha_hat * volatility_scaling
        
        n_param = 1.0 / alpha_dynamic
        p_param = n_param / (n_param + y_pred_final)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            lower_99 = nbinom.ppf(LCL, n_param, p_param)
            upper_99 = nbinom.ppf(UCL, n_param, p_param)
            
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
#============================================================================================================================
class ModelPredictorVelo:
    def __init__(self, model):
        # models è una lista [model_glm] come restituito da create_from_models
        if not model:
            raise ValueError("La lista di modelli è vuota.")
        self.model = model  # per GAUSSIAN si usa il primo (e unico) modello

    def predict(self, X_test, y_test, X_global, ds_test, df):

        # Z-score per confidenza al 99%
        z = 2.576
        model = self.model

        # 1. Previsione della media (mu)
        y_pred_final = model.predict(X_test).values

        # 2. Stima della deviazione standard (sigma)
        sigma_hat = np.sqrt(model.scale)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            # 3. Calcolo dei bound gaussiani: mu +/- z * sigma
            lower_99 = y_pred_final - (z * sigma_hat)
            upper_99 = y_pred_final + (z * sigma_hat)

            # 4. Post-processing: la velocità non può essere negativa
            lower_99 = np.maximum(0, lower_99)

            lower_99 = np.floor(np.nan_to_num(lower_99, nan=0.0))
            upper_99 = np.ceil(np.nan_to_num(upper_99, nan=np.max(y_pred_final) * 2))

        return pd.DataFrame({
            'ds':         ds_test.values,
            'y_real':     y_test.values,
            'y_pred':     y_pred_final,
            'lower_99':   lower_99,
            'upper_99':   upper_99,
            'model_sigma': sigma_hat
        })