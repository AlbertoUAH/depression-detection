# -- Libraries
from sklearn.model_selection import train_test_split
from pathlib import Path
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers.integrations import WandbCallback
from datasets import Dataset
from datasets import load_metric
import numpy as np
import pandas as pd
import torch
import params
import itertools
import glob
import wandb
import transformers
import os

# -- Constants
WANDB_PROJECT     = 'autextification'
RAW_DATA_AT       = 'raw_dataset'
PROCESSED_DATA_AT = 'raw_dataset_split'

# -- Setup TrainingArguments
training_args = TrainingArguments(
            report_to='wandb',
            output_dir="results",
            evaluation_strategy="epoch",
            metric_for_best_model="f1",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            num_train_epochs=1,
            weight_decay=0.01,
            fp16=True,
            load_best_model_at_end=True,  # Load best model once it has finished
            save_strategy='epoch',        # Save at each epoch
            save_total_limit=1            # Limit savings to 1
)

# -- Define custom metrics
accuracy_metric = load_metric("accuracy")
f1_metric       = load_metric("f1")
recall_metric   = load_metric("recall")
precision_metric= load_metric("precision")

# -- Download data and split train and val sets
def download_data(run):
    processed_data_at = run.use_artifact(f'{PROCESSED_DATA_AT}:latest')
    processed_dataset_dir = Path(processed_data_at.download())
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')
    df.reset_index(drop=True, inplace=True)
    print("COLUMNS: {}".format(df.columns))
    return df

def train_val_split(df):
    # Train & Test split
    df['label'] = df['label'].apply(lambda x: 1 if x == 'generated' else 0)
    train_df, val_df =  df[df['Stage'] == 'train'],\
                        df[df['Stage'] == 'val']

    train_df.drop('Stage', axis=1, inplace=True)
    val_df.drop('Stage', axis=1, inplace=True)
    return train_df, val_df

# -- Get model, tokenizer plus data
def get_model_tokenizer(model_name="xlm-roberta-base", device="cuda"):
    # Define model and tokenizer
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    return model, tokenizer

# Define tokenize function
def tokenize_data(data, tokenizer, device):
    texts = data["text"].tolist()
    labels = [int(x) for x in data["label"].tolist()] # Ajusta el nombre de la columna que contiene las etiquetas
    encoded_inputs = tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt").to(device)
    encoded_inputs["labels"] = torch.tensor(labels)
    return encoded_inputs

# -- Get validation logger
def get_validation_logger(val_dataset):
    validation_inputs  = val_dataset.remove_columns(['labels', 'attention_mask', 'input_ids'])
    validation_targets = [str(x) for x in val_dataset['labels']]

    validation_logger = ValidationDataLogger(
        inputs = validation_inputs[:],
        targets = validation_targets
    )
    return validation_logger

def compute_metrics_wandb(eval_pred, validation_logger):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # convert predictions from class (0, 1, 2…) to label (Health, Science…)
    prediction_labels = [int(x) for x in predictions]
    
    # log predictions
    validation_logger.log_predictions(prediction_labels)

    # metrics from the datasets library have a compute method
    f1 = f1_metric.compute(predictions=prediction_labels, references=labels)
    recall = recall_metric.compute(predictions=prediction_labels, references=labels)
    precision = precision_metric.compute(predictions=prediction_labels, references=labels)
    accuracy = accuracy_metric.compute(predictions=prediction_labels, references=labels)
    
    table = {
        'val_metrics_accuracy': accuracy,
        'val_metrics_precision': precision,
        'val_metrics_recall': recall,
        'val_metrics_f1': f1
    }
    wandb.log(table)
    return f1

def get_test_metrics(test_df, tokenizer, trainer, run, device):
    ids       = test_df['id'].tolist()
    topic_str = test_df['topic_str'].tolist()
    texts     = test_df['text'].apply(str).tolist()

    test_labels     = [int(x) for x in test_df['label'].tolist()]
    input_encodings = tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt").to(device)
    test_dataset    = Dataset.from_dict(input_encodings)
    
    predictions = trainer.predict(test_dataset)

    predicted_labels  = np.argmax(predictions.predictions, axis=1)
    predicted_labels  = [int(x) for x in predicted_labels]

    # metrics from the datasets library have a compute method
    f1 = f1_metric.compute(predictions=predicted_labels, references=test_labels)
    f1_micro = f1_metric.compute(predictions=predicted_labels, references=test_labels, average="micro")
    f1_macro = f1_metric.compute(predictions=predicted_labels, references=test_labels, average="macro")
    recall = recall_metric.compute(predictions=predicted_labels, references=test_labels)
    precision = precision_metric.compute(predictions=predicted_labels, references=test_labels)
    accuracy = accuracy_metric.compute(predictions=predicted_labels, references=test_labels)

    # -- Prediction table
    wandb_rows = list(zip(ids, topic_str, texts, test_labels, predicted_labels))
    wandb_rows = [list(row) for row in wandb_rows]
    preds_table = wandb.Table(
        columns=["id", "topic_str", "text", "original_label", "predicted_label"], 
        data=wandb_rows
    )
    wandb.log({"Test Dataset": preds_table})

    # -- Test metrics table
    test_row = [[f1, f1_micro, f1_macro, recall, precision, accuracy]]
    metrics_table = wandb.Table(
        columns=["test_f1", "test_micro_f1", "test_macro_f1", "recall", "precision", "accuracy"], 
        data=test_row
    )
    wandb.log(
        {
            'Test metrics': metrics_table

        })
    
    # Confussion Matrix
    test_labels_str = ["generated" if label else "human" for label in test_labels]
    predicted_labels_str = ["generated" if label else "human" for label in predicted_labels]
    # Confussion Matrix
    wandb.log({"test_confusion_matrix": wandb.sklearn.plot_confusion_matrix(y_true=test_labels_str, 
                y_pred=predicted_labels_str)})

# -- Function to save best model
def save_model_to_registry(run, folder_path, model_registry_path):
    # Save the model to W&B
    best_model = wandb.Artifact(f"model_{run.id}", type='model')
    best_model.add_dir(folder_path)
    run.log_artifact(best_model)

    # Link the model to the Model Registry
    run.link_artifact(best_model, model_registry_path)


# -- Define Trainer object
def train_model(training_args):
    # -- Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training_args")
    print(training_args)

    # -- Good practice to inject using sweeps
    run = wandb.init(project=WANDB_PROJECT, entity=params.ENTITY, 
                     job_type="training", config=training_args)

    wandb_config  = dict(wandb.config)
    model_name    = wandb_config['model']
    text_to_lowercase = wandb_config['text_to_lowercase']
    model_registry_path = wandb_config['model_registry_path']

    # -- Get test_path
    test_path = wandb_config['test_path']
    test_df   = pd.read_csv(test_path)

    # -- Remove unused parameters
    del wandb_config['distributed_state']
    del wandb_config['deepspeed_plugin']
    del wandb_config['model']
    del wandb_config['text_to_lowercase']
    del wandb_config['test_path']
    del wandb_config['model_registry_path']

    training_args = TrainingArguments(**wandb_config)
    
    # -- Download data
    df = download_data(run=run)
    print("Number of rows - train + val DataFrame: {}".format(df.shape[0]))
    print("Checking whether df has duplicated rows or not")
    df.drop_duplicates(subset=['text'], inplace=True)
    print("Number of rows after removing duplicates: {}".format(df.shape[0]))

    if text_to_lowercase > 0:
        df['text'] = df['text'].apply(lambda x: str(x).lower())
        test_df['text'] = test_df['text'].apply(lambda x: str(x).lower())
    
    # -- split into train and validation set
    train_df, val_df = train_val_split(df)
    
    # -- Get model and tokenizer
    model, tokenizer = get_model_tokenizer(model_name=model_name, device=device)
    
    # -- Tokenize data
    train_encoded = tokenize_data(train_df, tokenizer, device)
    val_encoded   = tokenize_data(val_df, tokenizer, device)
    # -- Convert codified data into Dataset format
    train_dataset = Dataset.from_dict(train_encoded)
    val_dataset   = Dataset.from_dict(val_encoded)
    
    # -- Get validation logger
    validation_logger = get_validation_logger(val_dataset)
    
    extra_args = {
        'validation_logger': validation_logger
    }
    
    # -- Define Trainer class
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda eval_pred: compute_metrics_wandb(eval_pred, **extra_args)
    )
    # -- Train
    trainer.train()
    print("Training process finished!")
    get_test_metrics(test_df, tokenizer, trainer, run, device)
    print("Testing process finished!")

    # -- Finally, save model and finish wandb run
    folder_path = glob.glob('/kaggle/working/results/checkpoint*')[0] + '/'
    save_model_to_registry(run, folder_path, model_registry_path)
    run.finish()
    print("Finished!")

if __name__ == '__main__':
    train_model(training_args)
