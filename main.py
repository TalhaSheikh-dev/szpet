from scrapper.working import id_scrapper
from src.utils import dataframe_to_jsonl, format_df_for_pet_pattern
from pathlib import Path
import os
from src.train import train
from src.config import Config

import torch
from tqdm.notebook import tqdm
from src.eval import load_model
from src.utils import batch_data_from_dataframe
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def print_evaluation(df_test):
    batches = batch_data_from_dataframe(df_test, desc_cols=['text'], batch_size=4)

    outputs = []
    logits = []
    with torch.no_grad():
        for batch in tqdm(batches):
            pred_lbl, lbl_logits = model.predict(batch)
            logits.extend(lbl_logits.cpu().numpy().tolist())

            outputs.extend(pred_lbl.cpu().numpy().tolist())
    df_test['machine_label'] = ['positive' if x == 1 else "negative" for x in outputs]
    true_label = df_test["label"]
    
    prediction = df_test["machine_label"]

    print(classification_report(true_label, prediction))
    cm = confusion_matrix(true_label, prediction, labels=["negative", "positive"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "positive"])
    disp.plot()
    plt.show()

    return df_test



#df_test = id_scrapper()
import pandas as pd
df_test = pd.read_csv("/home/talhasheikh/Documents/wavo_co_scraper/data_replies.csv")
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


os.environ['SZ_ROOT'] = '/home/talhasheikh/Documents/sale/email_classification/szpet'
config = Config(kwargs=config, mkdir=True)


model_path = 'exp_out/generic/facebook/muppet-roberta-large/2022-05-26-21-36-28/best_model.pt'

model = load_model(config, model_path)
model.eval()

output = print_evaluation(df_test)
output.to_csv("evaluation/test.csv", index=False)



