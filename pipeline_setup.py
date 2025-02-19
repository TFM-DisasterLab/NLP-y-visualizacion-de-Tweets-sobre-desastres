import os
import argparse
import pandas as pd
import papermill as pm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import torch
import time

def loading_message(message):
    """Muestra un mensaje de progreso con puntos animados."""
    print(message, end='', flush=True)
    for _ in range(3):
        time.sleep(0.5)
        print('.', end='', flush=True)
    print(" done.")

def clean_up(files):
    for file in files:
        try:
            os.remove(file)
            print(f"Removed file: {file}")
        except Exception as e:
            print(f"Error removing file {file}: {e}")

def run_notebook(input_notebook: str, output_notebook: str):
    loading_message(f"Running notebook: {input_notebook}")
    pm.execute_notebook(input_notebook, output_notebook)
    loading_message(f"Notebook executed: {input_notebook}")

def load_dataframe(csv_path: str):
    loading_message(f"Loading CSV: {csv_path}")
    return pd.read_csv(csv_path)

def load_model(model_path: str):
    loading_message(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    return tokenizer, model

def classify_model(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).tolist()
    return predictions

def load_label_encoder(model_path: str):
    loading_message(f"Loading label encoder from: {model_path}")
    label_encoder_path = f"{model_path}/label_encoder.pkl"
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder

def decode_labels(predictions, label_encoder):
    if hasattr(label_encoder, "inverse_transform"):
        return label_encoder.inverse_transform(predictions)
    try:
        inverse_mapping = {v: k for k, v in label_encoder.items()}
        return [inverse_mapping[i] for i in predictions]
    except AttributeError:
        return [label_encoder[i] for i in predictions]

def main():
    loading_message("Starting pipeline")
    
    # notebook de la API de Twitter (genera tweets.csv)
    twitter_api_input = "Twitter_API.ipynb"
    twitter_api_output = "Twitter_API_output.ipynb"
    run_notebook(twitter_api_input, twitter_api_output)
    
    # preprocess notebook (usa tweets.csv y genera disaster_preprocessed.csv)
    preprocessing_input_notebook = "PREPROCESSING_API_FINAL.ipynb"
    preprocessing_output_notebook = "PREPROCESSING_API_FINAL_output.ipynb"
    run_notebook(preprocessing_input_notebook, preprocessing_output_notebook)
    
    #  df (disaster_preprocessed.csv)
    preprocessed_csv = "disaster_preprocessed.csv"
    df_preprocessed = load_dataframe(preprocessed_csv)
    #
    texts = df_preprocessed["Text"].tolist()

    # 4. path modelos
    model1_path = "models/bert_informative_classifier_1"
    model2_path = "models/bert_event_classifier_2"
    model3_path = "models/bert_information_classifier_3"
    
    #  modelos
    tokenizer1, model1 = load_model(model1_path)
    tokenizer2, model2 = load_model(model2_path)
    tokenizer3, model3 = load_model(model3_path)
    
    # label encoders
    label_encoder1 = load_label_encoder(model1_path)
    label_encoder2 = load_label_encoder(model2_path)
    label_encoder3 = load_label_encoder(model3_path)
    
    #Pipeline de modelos
    
    # Modelo 1
    loading_message("Running Model 1 predictions")
    predictions1 = classify_model(texts, tokenizer1, model1)
    labels1 = decode_labels(predictions1, label_encoder1)
    
    # Modelo 2: Ejecutar para TODOS los tweets
    loading_message("Running Model 2 predictions on all tweets")
    predictions2 = classify_model(texts, tokenizer2, model2)
    labels2_all = decode_labels(predictions2, label_encoder2)
    
    model2_labels = []
    for l1, l2 in zip(labels1, labels2_all):
        if l1.lower() == "informative":
            model2_labels.append(l2)
        else:
            if l2 and l2.strip() != "":
                model2_labels.append(f"{l2} (not informative)")
            else:
                model2_labels.append("Not Available")
    
    # Modelo 3
    loading_message("Running Model 3 predictions")
    predictions3 = classify_model(texts, tokenizer3, model3)
    labels3 = decode_labels(predictions3, label_encoder3)
    
    
    df_preprocessed["Model1"] = labels1
    df_preprocessed["Model2"] = model2_labels
    df_preprocessed["Model3"] = labels3
    df_preprocessed = df_preprocessed.fillna("N/A")

    # verificacion de duplicados check 
    loading_message("Verificando duplicados y guardando resultados")
    output_csv = "dataframe_api_historico.csv"
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        df_combined = pd.concat([df_existing, df_preprocessed], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["AuthorID", "Text"], keep="first")
        df_combined.to_csv(output_csv, index=False)
        print(f"Updated combined results saved to {output_csv}")
    else:
        df_preprocessed.to_csv(output_csv, index=False)
        print(f"Saved combined results to {output_csv}")
    
    # borrar archivos innecesarios
    files_to_remove = [
        "PREPROCESSING_API_FINAL_output.ipynb",
        "Twitter_API_output.ipynb",
        "disaster_preprocessed.csv"
    ]
    clean_up(files_to_remove)
    loading_message("Pipeline completed")

if __name__ == "__main__":
    main()








