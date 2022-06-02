from scrapper.working import id_scrapper
from src.utils import dataframe_to_jsonl, format_df_for_pet_pattern
from pathlib import Path
import os
from src.train import train
from src.config import Config

import torch
from src.eval import load_model
from src.utils import batch_data_from_dataframe
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def print_evaluation(df_test):
    batches = batch_data_from_dataframe(df_test, desc_cols=['text'], batch_size=4)

    outputs = []
    logits = []
    with torch.no_grad():
        for batch in batches:
            pred_lbl, lbl_logits = model.predict(batch)
            logits.extend(lbl_logits.cpu().numpy().tolist())

            outputs.extend(pred_lbl.cpu().numpy().tolist())
    df_test['machine_label'] = ['positive' if x == 1 else "negative" for x in outputs]
    return df_test



df_test = id_scrapper()
import pandas as pd
#df_test = pd.read_csv("/home/talhasheikh/Documents/wavo_co_scraper/data_replies.csv")
df_test = df_test.sample(frac=1).reset_index(drop=True)



config = {
    "pretrained_weight":  "facebook/muppet-roberta-large",
    "dataset": "generic",
    "data_dir": "data/szinternal/",
    "pattern": "Question: What is the sentiment of this email:[TEXT1]\nAnswer (positive/negative): [LBL]. [SEP]",
    "dict_verbalizer": {'negative': 'negative', 'positive': "positive"},
    "pattern_idx": 1,
    "idx_txt_trim": 1,
    "max_text_length": 256,
    "batch_size": 4,
    "eval_batch_size": 4,
    "num_batches": 25,
    "max_num_lbl_tok": 3,
    "eval_every": 50,
    "eval_train": True,
    "warmup_ratio": 0.06,
    "mask_alpha": 0.15,
    "grad_accumulation_factor": 4,
    "seed": 42,
    "lr": 1e-6,
    "weight_decay": 1e-2
}


os.environ['SZ_ROOT'] = '/home/talhasheikh/szpet'
config = Config(kwargs=config, mkdir=True)


model_path = '/home/talhasheikh/szpet/weights/best_model.pt'

model = load_model(config, model_path)
model.eval()

output = print_evaluation(df_test)
output = output[["text","machine_label"]]
output.to_csv("gs://szmodels/email_classification_model/test.csv", index=False)


import sys
import typing
import google.cloud.compute_v1 as compute_v1

def stop_instance(project_id, zone, machine_name):
    instance_client = compute_v1.InstancesClient()
    operation_client = compute_v1.ZoneOperationsClient()

    print(f"Stopping {machine_name} from {zone}...")
    operation = instance_client.stop(
        project=project_id, zone=zone, instance=machine_name
    )
    while operation.status != compute_v1.Operation.Status.DONE:
        operation = operation_client.wait(
            operation=operation.name, zone=zone, project=project_id
        )
    if operation.error:
        print("Error during stop:", operation.error, file=sys.stderr)
    if operation.warnings:
        print("Warning during stop:", operation.warnings, file=sys.stderr)
    print(f"Instance {machine_name} stopped.")
    return

stop_instance('szlm-333022', 'us-central1-c', 'talha2')
