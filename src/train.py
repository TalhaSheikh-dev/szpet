import torch
import os
import numpy as np
import argparse
import logging
import transformers as tfrms

from src.eval import dev_eval
from src.szpet import adapet
from torch.optim.lr_scheduler import LambdaLR

from src.batcher import Batcher
from src.utils import get_avg_dict_val_store, update_dict_val_store, device
from tqdm.notebook import tqdm

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(config):
    '''
    Trains the model
    :param config:
    :return:
    '''

    tokenizer = tfrms.AutoTokenizer.from_pretrained(config.pretrained_weight)
    batcher = Batcher(config, tokenizer, config.dataset)
    dataset_reader = batcher.get_dataset_reader()
    print('Initializing Model\n')
    model = adapet(config, tokenizer, dataset_reader).to(device)

    ### Create Optimizer
    # Ignore weight decay for certain parameters
    no_decay_param = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay_param)],
         'weight_decay': config.weight_decay,
         'lr': config.lr},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay_param)],
         'weight_decay': 0.0,
         'lr': config.lr},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-8)

    best_dev_acc = 0
    train_iter = batcher.get_train_batch()
    dict_val_store = None

    # Number of batches is assuming grad_accumulation_factor forms one batch
    tot_num_batches = config.num_batches * config.grad_accumulation_factor

    # Warmup steps and total steps are based on batches, not epochs
    num_warmup_steps = config.num_batches * config.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, config.num_batches)

    for i in tqdm(range(tot_num_batches), desc='Train Batches'):
        # Get true batch_idx
        batch_idx = (i // config.grad_accumulation_factor)

        model.train()
        sup_batch = next(train_iter)
        loss, dict_val_update = model(sup_batch)
        loss = loss / config.grad_accumulation_factor
        loss.backward()

        if (i+1) % config.grad_accumulation_factor == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        dict_val_store = update_dict_val_store(dict_val_store, dict_val_update, config.grad_accumulation_factor)
        print("Finished %d batches" % batch_idx, end='\r')

        if (batch_idx + 1) % config.eval_every == 0 and i % config.grad_accumulation_factor == 0:
            dict_avg_val = get_avg_dict_val_store(dict_val_store, config.eval_every)
            dict_val_store = None
            dev_acc, dev_logits = dev_eval(config, model, batcher, batch_idx, dict_avg_val)

            print("Global Step: %d Acc: %.3f" % (batch_idx, dev_acc) + '\n')
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                torch.save(model.state_dict(), os.path.join(config.exp_dir, "best_model.pt"))
                with open(os.path.join(config.exp_dir, "dev_logits.npy"), 'wb') as f:
                    np.save(f, dev_logits)