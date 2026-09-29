#import
import sys
import os

# Risale di un livello per trovare la cartella radice del progetto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Nasconde messaggi INFO e WARNING
import pandas as pd
import pickle
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

prov = "MI".upper()
com = "F205".upper()
zona = "R2".upper()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROV_DIR = os.path.join(ROOT_DIR, "output_province")
FILE_PATH = os.path.join(ROOT_DIR, "output_province", f"traffico_{prov}.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "Modelli_traffico", prov)

os.makedirs(OUTPUT_DIR, exist_ok=True)

parq = pd.read_parquet(FILE_PATH)
to_prep = parq[(parq['comune_amm'] == com.upper()) & (parq['zona'] == zona.upper())]
df          = final_preprocess(to_prep)
base, info  = create_all_base_features(df, k_hour=4, k_week=2)
inter, _    = create_interactions(base, info)
X_global    = pd.concat([base, inter], axis=1)
y           = df['count_traffico']

start_train = pd.to_datetime("2024-01-01 00:00:00")
end_train = pd.to_datetime("2024-12-31 23:00:00")

start_test = pd.to_datetime("2026-01-16 00:00:00")
end_test =  pd.to_datetime("2026-01-25 23:00:00")

X_global = X_global.copy()
X_train = X_global[df['datetime'].between(start_train, end_train)]
y_train = df[df['datetime'].between(start_train, end_train)]['count_traffico']
print(f"Dimensioni df TRAIN con armoniche: {X_train.shape}, dimensioni serie storica: {y_train.shape}")

X_global = X_global.copy()
X_test = X_global[df['datetime'].between(start_test, end_test)]
y_test = df[df['datetime'].between(start_test, end_test)]['count_traffico']
ds_test = df[df['datetime'].between(start_test, end_test)]['datetime']
print(f"Dimensioni df TRAIN con armoniche: {X_test.shape}, dimensioni serie storica: {y_test.shape}")

model_name, model = create_from_models(
                        X_train = X_train,
                        y_train = y_train,
                        model_type = 'negbin+sarimax')

my_model = ModelPredictor(
                        model_nb = model[0],
                        model_sarima= model[1],
                        alpha = model[0].params['alpha'],
                        )

nome_modello = f'{prov}_{com}_{zona}_{model_name}.pkl'

OUTPUT_FILE = os.path.join(OUTPUT_DIR, nome_modello)

joblib.dump(my_model, OUTPUT_FILE, compress=True)

print(f'Classe ModelPredictor per {prov}, {com}, {zona} salvata con successo')