import numpy as np
import statsmodels.api as sm

#===================================================================================================================================================

def create_ols_model(y_train, X_train):
    # 1. Fit nel dominio log
    models = []
    y_train_log = np.log1p(y_train)
    model_ols = sm.OLS(y_train_log, X_train).fit()
    models.append(model_ols)
    
    return models

#===================================================================================================================================================

def create_negbin_sarimax_model(y_train, X_train):
    """
    Logica Ibrida: Negative Binomial per la componente regressiva + SARIMA sui residui.
    """
    import pandas as pd
    import warnings
    print(f'print da dentro create_logiche {X_train.shape}')
    models = []
    
    # NB fit con suppression del convergence warning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=sm.tools.sm_exceptions.ConvergenceWarning)
        model_nb = sm.NegativeBinomial(y_train, X_train).fit_regularized(
            disp=False,
            warn_convergence=False
        )
    models.append(model_nb)
    mu_train_nb = model_nb.predict(X_train)
    train_resid = y_train - mu_train_nb
    
    # Assicura che train_resid abbia un DatetimeIndex valido per SARIMAX
    if not isinstance(train_resid.index, pd.DatetimeIndex):
        # Se l'indice non è DatetimeIndex, usa un indice numerico per evitare il warning
        train_resid_for_sarima = train_resid.reset_index(drop=True)
    else:
        train_resid_for_sarima = train_resid
    
    # SARIMAX fit con suppression del convergence warning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=sm.tools.sm_exceptions.ConvergenceWarning)
        sarima_res = sm.tsa.SARIMAX(
            train_resid_for_sarima, 
            order=(0, 0, 0), 
            seasonal_order=(1, 0, 0, 24),
            enforce_stationarity=False
        ).fit(disp=False, warn_convergence=False)
    #ricordare di aggiungere residui di una settimana prima quando si manda in previsione
    models.append(sarima_res)
    return models

#===================================================================================================================================================

def create_negbin_model(y_train, X_train):
    """
    Logica Negative Binomial corretta: 
    Usa sm.NegativeBinomial per stimare correttamente l'alpha (overdispersion).
    """
    import warnings
    models = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=sm.tools.sm_exceptions.ConvergenceWarning)
        model_nb = sm.NegativeBinomial(y_train, X_train).fit_regularized(
            disp=False,
            warn_convergence=False
        )
    models.append(model_nb)
    return models
#===================================================================================================================================================
def create_gaussian_glm_model(y_train, X_train):
    """
    Logica Gaussiana (Normale): 
    Utilizza un Modello Lineare Generalizzato (GLM) con famiglia Gaussian.
    """
    import warnings
    models = []
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=sm.tools.sm_exceptions.ConvergenceWarning)
        
        # Utilizziamo la famiglia Gaussian. 
        # Il link di default è Identity (y = X*beta), 
        # ma puoi usare link=sm.families.links.Log() se vuoi garantire velocità sempre positive.
        model_gauss = sm.GLM(
            y_train, 
            X_train, 
            family=sm.families.Gaussian()
        ).fit()
        
    models.append(model_gauss)
    return models