#this file is for the creation of all the forecasting models. it's long to run and probably not worth the time.
#considering it's just a loop of what happens in the "single model" file, i suggest skipping this file and go to the next.
#to see graphed the results of the model created. I suggest to start looking at "create_predict_model_final.ipynb".

import sys
import os

# Risale di un livello per trovare la cartella radice del progetto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Nasconde messaggi INFO e WARNING
import pandas as pd
import joblib
import os
import warnings
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")
from richieste.funzioni_commit import (
                                create_all_base_features,
                                create_interactions,
                                final_preprocess,
                                create_from_models
)

from richieste.classes import ModelPredictor

def ensure_const_and_align(X, const_name="const"):
    X = X.copy()

    # 1. Aggiunta esplicita della costante se manca
    if const_name not in X.columns:
        X.insert(0, const_name, 1.0)
    return X
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROV_DIR = os.path.join(ROOT_DIR, "output_province")
print("PATH TROVATO:", PROV_DIR)
START_TRAIN = pd.to_datetime("2024-01-01 00:00:00")
END_TRAIN = pd.to_datetime("2024-12-31 23:00:00")

for filename in os.listdir(PROV_DIR):
    if filename.endswith("PO.parquet"):
        
        # fname es. "traffico_MI.parquet" oppure "traffico_CZ.parquet"
        prov = os.path.splitext(filename)[0].split("_", 1)[1]
        # qui prov == "MI" o "CZ" …
        FILE_PATH = os.path.join(ROOT_DIR, "output_province", f"traffico_{prov}.parquet")
        OUTPUT_DIR = os.path.join(BASE_DIR, "Modelli_traffico", prov)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        parq = pd.read_parquet(FILE_PATH)
        for com in parq.comune_amm.unique():
            zone_groups = parq[parq['comune_amm'] == com]['zona'].unique()
            for zona in zone_groups:
                    
                    print(f'\nmomentaneamente calcolando il modello per: prov {prov}, comune {com}, zona {zona}')
                    to_prep = parq[(parq['comune_amm'] == com.upper()) & (parq['zona'] == zona.upper())]
                    df          = final_preprocess(to_prep)
                    base, info  = create_all_base_features(df, k_hour=4, k_week=2)
                    inter, _    = create_interactions(base, info)
                    X_global    = pd.concat([base, inter], axis=1)
                    y           = df['count_traffico']

                    X_global = X_global.copy()
                    X_train = X_global[df['datetime'].between(START_TRAIN, END_TRAIN)]
                    y_train = df[df['datetime'].between(START_TRAIN, END_TRAIN)]['count_traffico']
                    print(f"Dimensioni df TRAIN con armoniche: {X_train.shape}, dimensioni serie storica: {y_train.shape}")

                    try:
                        model_name, model = create_from_models(
                                                X_train = X_train,
                                                y_train = y_train,
                                                #sono abbastanza sicuro che usare qualsiasi nome diverso da "negbin+sarimax" dia errore in quanto non ho 
                                                #ancora programmato quella parte di logica. comunque performano peggio di negbin su sarimax quindi poco male
                                                model_type = 'negbin+sarimax')
                    except Exception as exc:
                        print(f"ERROR: creazione modello fallita per {prov}, {com}, {zona} - {exc}")
                        continue

                    # controlliamo che il modello NB sia usabile
                    if not model or model[0] is None:
                        print(f"Modello NB non disponibile per {prov}, {com}, {zona}, salto")
                        continue

                    my_model = ModelPredictor(model_nb = model[0],
                                            model_sarima= model[1],
                                            alpha = getattr(model[0].params, 'alpha', None)
                                            )

                    nome_modello = f'{prov}_{com}_{zona}_{model_name}.pkl'

                    OUTPUT_FILE = os.path.join(OUTPUT_DIR, nome_modello)

                    joblib.dump(my_model, OUTPUT_FILE, compress=9)

                    print(f'Classe ModelPredictor per {prov}, {com}, {zona} salvata con successo')
                    print(f"Codice completato alle: {pd.Timestamp.now()}")
