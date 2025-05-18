from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments #Import TrainingArguments instead of TrainingArgument
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Chargement des deux datasets annotés pour le sentiment financier
ds1 = load_dataset("zeroshot/twitter-financial-news-sentiment")
ds2 = load_dataset("nickmuchi/financial-classification")

print(ds1) #train et validation
print(ds2) #train et test

# On harmonise les labels des deux jeux de données d'entraînement de sentiment financier provenant de sources différentes.
# Pour chaque dataset, les labels d'origine (bearish, bullish, neutral ou negative, positive, neutral) sont remappés afin d'assurer une cohérence d'encodage entre les jeux de données.
# On applique ensuite ces fonctions de mapping à chaque dataset, puis on uniformise le nom de la colonne cible et on supprime les colonnes redondantes si nécessaire.

def map_labels(example):
    # Pour ds1 :
    # Check if the key '__hf_source' exists before accessing it
    if '__hf_source' in example and "twitter-financial-news-sentiment" in str(example['__hf_source']):
        if example['label'] == 0: # Bearish
            example['label'] = 2 # Negative
        elif example['label'] == 1: # Bullish
            example['label'] = 1 # Positive
        elif example['label'] == 2: # Neutral
            example['label'] = 0 # Neutral
    return example

ds1 = ds1.map(map_labels)

def map_labels2(example):
    # Pour ds2 :
    # Check if '__hf_source' exists before accessing it
    if '__hf_source' in example and "financial-classification" in str(example['__hf_source']):
        # This block was empty, adding a pass statement to avoid syntax error
        pass
    #If '__hf_source' is not present, we assume the example is from financial-classification (since it's being applied to ds2)
    else: # This else statement was not indented correctly
        if example['labels'] == 0: # Negative
            example['labels'] = 2 # Negative
        elif example['labels'] == 1: # Positive
            example['labels'] = 1 # Positive
        elif example['labels'] == 2: # Neutral
            example['labels'] = 0 # Neutral
    return example

ds2 = ds2.map(map_labels2)


# Renommer 'labels' en 'label' dans ds2 après avoir fait la conversion :
ds2 = ds2.rename_column("labels", "label")

# Supprimer la colonne labels si elle existe :
try:
    ds1 = ds1.remove_columns("labels")
except:
    pass

# 2. Concaténation des datasets
combined_dataset = DatasetDict({
    'train': concatenate_datasets([ds1['train'], ds2['train']]),
    'test': concatenate_datasets([ds1['validation'], ds2['test']])
})

print("Dataset combiné :")
print(combined_dataset)

!pip install --upgrade transformers  # si nécessaire

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os

def train_model(model_name, dataset, batch_size=16, num_epochs=3, save_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = dataset['train'].map(tokenize_function, batched=True)
    tokenized_test = dataset['test'].map(tokenize_function, batched=True)

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=f"./{model_name.replace('/', '_')}_results",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=f"./{model_name.replace('/', '_')}_logs",
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()

    # ICI : chemin de sauvegarde sur ton ordinateur Mac
    if save_path is None:
        base_path = "./models"
        save_path = os.path.join(base_path, f"{model_name.replace('/', '_')}_finetuned")

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"✅ Modèle '{model_name}' entraîné et sauvegardé dans : {save_path}")
    print(f"📊 Performances : {metrics}")
    return metrics

# Finetuner les modèles BERT et FinBERT, comparer les performances et sauvegarder les poids pour le TP8.
train_model("bert-base-uncased", combined_dataset, batch_size=16, num_epochs=1)
train_model("yiyanghkust/finbert-tone", combined_dataset, batch_size=16, num_epochs=1)
