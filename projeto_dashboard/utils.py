import pandas as pd
import numpy as np
from datetime import timedelta

def carregar_dados(caminho_arquivo):
    df = pd.read_excel(caminho_arquivo, sheet_name='Folha1')
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = [
        "Index", "Video Time", "Atleta", "Neutral", "Happy", "Sad", "Angry",
        "Surprised", "Scared", "Disgusted", "Valence", "Arousal"
    ]
    df = df.drop(columns=["Index"])

    emotion_cols = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Scared", "Disgusted", "Valence", "Arousal"]
    df = df[~df[emotion_cols].apply(lambda row: row.astype(str).str.contains("FIT_FAILED")).any(axis=1)]
    df[emotion_cols] = df[emotion_cols].astype(float)

    df["Video Time"] = pd.to_datetime(df["Video Time"], format="%H:%M:%S.%f").dt.time
    df["Video Time"] = df["Video Time"].apply(lambda t: timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond))

    intervalos = [
        (timedelta(seconds=start), timedelta(seconds=end, milliseconds=800))
        for start, end in [
            (6, 14), (29, 37), (52, 60), (75, 83), (98, 105), (121, 129),
            (144, 152), (167, 175), (190, 198), (213, 221), (237, 244), (260, 268)
        ]
    ]

    def atribuir_intervalo(tempo):
        for i, (inicio, fim) in enumerate(intervalos, 1):
            if inicio <= tempo <= fim:
                return f"Intervalo_{i}"
        return "Fora_Intervalo"

    df["Intervalo"] = df["Video Time"].apply(atribuir_intervalo)
    return df

def carregar_inqueritos(caminho_arquivo):
    df_inq = pd.read_excel(caminho_arquivo, sheet_name="inqueritos")
    return df_inq.rename(columns={
        "Valence normalizado": "Valence_Inquerito",
        "Arousal normalizado": "Arousal_Inquerito"
    })

def carregar_oasis(caminho_arquivo):
    df_oasis = pd.read_excel(caminho_arquivo, sheet_name="OASIS_ref") 
    df_oasis["Imagem"] = df_oasis["Imagem"].astype(str).str.strip()
    return df_oasis

def normalizar_oasis(df):
    df["Valence_OASIS"] = (df["Valence_OASIS"] - 4) / 3
    df["Arousal_OASIS"] = (df["Arousal_OASIS"] - 1) / 6
    return df

def calcular_face_reader_medias(df):
    df_intervalado = df[df["Intervalo"] != "Fora_Intervalo"].copy()
    df_intervalado["Imagem"] = df_intervalado["Intervalo"].apply(lambda x: int(x.split("_")[1]))
    df_intervalado = df_intervalado.rename(columns={"Atleta": "Jogadora"})
    return df_intervalado.groupby(["Jogadora", "Imagem"])[["Valence", "Arousal"]].mean().reset_index()

def comparar_face_reader_vs_inquerito(df_dashboard, df_inq):
    df_face = calcular_face_reader_medias(df_dashboard)
    comparado = pd.merge(df_face, df_inq, on=["Jogadora", "Imagem"])
    return comparado

def adicionar_distancia_emocional(df):
    df["Dist_Valence"] = (df["Valence"] - df["Valence_Inquerito"])**2
    df["Dist_Arousal"] = (df["Arousal"] - df["Arousal_Inquerito"])**2
    df["Dist_Euclidiana"] = (df["Dist_Valence"] + df["Dist_Arousal"])**0.5
    return df

def calcular_distancias_ao_oasis(df_comparado):
    df_comparado["Dist_FR_OASIS"] = np.sqrt(
        (df_comparado["Valence"] - df_comparado["Valence_OASIS"])**2 +
        (df_comparado["Arousal"] - df_comparado["Arousal_OASIS"])**2
    )
    df_comparado["Dist_INQ_OASIS"] = np.sqrt(
        (df_comparado["Valence_Inquerito"] - df_comparado["Valence_OASIS"])**2 +
        (df_comparado["Arousal_Inquerito"] - df_comparado["Arousal_OASIS"])**2
    )
    return df_comparado

def definir_quadrante(valence, arousal):
    if valence >= 0 and arousal >= 0.5:
        return "Q1"
    elif valence < 0 and arousal >= 0.5:
        return "Q2"
    elif valence < 0 and arousal < 0.5:
        return "Q3"
    else:  # valence >= 0 and arousal < 0.5
        return "Q4"