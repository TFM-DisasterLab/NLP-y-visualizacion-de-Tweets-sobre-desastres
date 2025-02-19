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

def load_model(model_path: str, device):
    loading_message(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.to(device) 
    return tokenizer, model

def classify_model_in_batches(texts, tokenizer, model, device, batch_size=30):
    """Processes texts in smaller batches to reduce GPU memory usage."""
    all_predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).tolist()
        all_predictions.extend(preds)
        torch.cuda.empty_cache()
    return all_predictions

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
    # acctivamos cuda cores (si estan disponibles)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loading_message("Starting pipeline")
    
    twitter_api_input = "Twitter_API.ipynb"
    twitter_api_output = "Twitter_API_output.ipynb"
    run_notebook(twitter_api_input, twitter_api_output)
    
    # 
    preprocessing_input_notebook = "PREPROCESSING_API_FINAL.ipynb"
    preprocessing_output_notebook = "PREPROCESSING_API_FINAL_output.ipynb"
    run_notebook(preprocessing_input_notebook, preprocessing_output_notebook)
    
    # cargamos el dataframe de desastres
    preprocessed_csv = "disaster_preprocessed.csv"
    df_preprocessed = load_dataframe(preprocessed_csv)
    
    
    texts = df_preprocessed["Text"].tolist()

    # path
    model1_path = "models/bert_informative_classifier_1"
    model2_path = "models/bert_event_classifier_2"
    model3_path = "models/bert_information_classifier_3"
    
    #Cargamos modelos en el dispositivo de prefedrencia (en nuestro caso es una 4080 super 16gb VRAM)
    tokenizer1, model1 = load_model(model1_path, device)
    tokenizer2, model2 = load_model(model2_path, device)
    tokenizer3, model3 = load_model(model3_path, device)
    
    # label encoders
    label_encoder1 = load_label_encoder(model1_path)
    label_encoder2 = load_label_encoder(model2_path)
    label_encoder3 = load_label_encoder(model3_path)
    
    # Modelo pipeline
    loading_message("Running Model 1 predictions")
    predictions1 = classify_model_in_batches(texts, tokenizer1, model1, device)
    labels1 = decode_labels(predictions1, label_encoder1)
    
    loading_message("Running Model 2 predictions on all tweets")
    predictions2 = classify_model_in_batches(texts, tokenizer2, model2, device)
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
    
    loading_message("Running Model 3 predictions")
    predictions3 = classify_model_in_batches(texts, tokenizer3, model3, device)
    labels3 = decode_labels(predictions3, label_encoder3)
    
    df_preprocessed["Model1"] = labels1
    df_preprocessed["Model2"] = model2_labels
    df_preprocessed["Model3"] = labels3
    df_preprocessed = df_preprocessed.fillna("N/A")

    # duplicados checkerr
    loading_message("Verifying duplicates and saving results")
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
        
    dashboard_input = "DASHBOARD_TFM.ipynb"
    dashboard_output = "DASHBOARD_TFM_output.ipynb"
    loading_message("Running dashboard notebook to refresh dashboard")
    run_notebook(dashboard_input, dashboard_output)
    
    # limpia
    files_to_remove = [
        "PREPROCESSING_API_FINAL_output.ipynb",
        "Twitter_API_output.ipynb",
        "disaster_preprocessed.csv",
        "DASHBOARD_TFM_output.ipynb"
    ]
    clean_up(files_to_remove)
    loading_message("Pipeline completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the disaster pipeline. Use --loop to run it 7 times in a loop.")
    parser.add_argument("--loop", action="store_true", help="If set, run the pipeline 7 times")
    args = parser.parse_args()

    if args.loop:
        for i in range(15):
            print(f"\n--- Starting pipeline iteration {i+1}/15 ---")
            main()
            print(f"--- Completed iteration {i+1}/15 ---\n")
        print("All iterations completed.")
    else:
        main()




