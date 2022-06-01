import os
import json
import random
import itertools
from collections import defaultdict
import numpy as np
import torch
from torch.utils import data
from src.tokenize import tokenize_pet_txt, tokenize_pet_mlm_txt
from src.utils import device


class Dataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, get_idx):
        return self.data[get_idx]

# Note that we can't run on 'test' data without a "flush_file" method
class DatasetReader(object):
    '''
    DatasetReader reads in a dataset
    '''

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer


        self.num_lbl = len(self.config.dict_verbalizer)

        self.check_pattern(self.config.pattern)

        txt_idx_trim = "[TEXT%d]" % self.config.idx_txt_trim

        self.pattern = self.config.pattern.split(txt_idx_trim)
        self.pattern.insert(1, txt_idx_trim)

        self.label = list(self.config.dict_verbalizer.values())

    def check_pattern(self, pattern):

        # Get maximum number of text
        self.text_ctr = 1
        while True:
            text_str = "[TEXT%d]" % self.text_ctr
            if text_str in pattern:
                self.text_ctr +=1
            else:
                break

        if self.text_ctr == 1:
            raise ValueError("Need at least one text ")

        if self.config.idx_txt_trim > self.text_ctr:
            raise ValueError("Text idx to trim %d is larger than number of text inputs %d" % (self.config.idx_txt_trim, self.text_ctr))

        num_mask_tok = pattern.count("[LBL]")
        if num_mask_tok != 1:
            raise ValueError("[LBL] must be in pattern 1 time, but is in pattern %d times" % pattern)


    def _get_file(self, split):
        '''
        Get filename of split
        :param split:
        :return:
        '''
        if split.lower() == "train":
            file = os.path.join(self.config.data_dir, "train.jsonl")
        elif split.lower() == "dev":
            file = os.path.join(self.config.data_dir, "val.jsonl")
        elif split.lower() == "test":
            file = os.path.join(self.config.data_dir, "test.jsonl")
        return file

    def get_num_lbl_tok(self):
        return self.config.max_num_lbl_tok

    def get_num_lbl(self):
        '''
        Get number of lbls in dataset
        :return:
        '''
        return self.num_lbl

    def read_dataset(self, split=None, is_eval=None):
        '''
        Read the dataset
        :param split: partition of the dataset
        :param is_eval:
        :return:
        '''
        file = self._get_file(split)

        data = []

        with open(file, 'r') as f_in:
            for i, line in enumerate(f_in.readlines()):
                json_string = json.loads(line)

                dict_input = {}
                dict_input["idx"] = i
                for j in range(1, self.text_ctr):
                    dict_input["TEXT%d" % j] = json_string["TEXT%d" % j]

                dict_output = {}
                if "LBL" not in json_string:
                    raise ValueError("LBL not in json")

                if json_string["LBL"] not in self.config.dict_verbalizer:
                    raise ValueError("Label %s not in dictionary verbalizer" % json_string["LBL"])
                dict_output["lbl"] = list(self.config.dict_verbalizer.keys()).index(json_string["LBL"])
                dict_input_output = {"input": dict_input, "output": dict_output}
                data.append(dict_input_output)
        return np.asarray(data)

    @property
    def pets(self):
        return ["PET1"]

    def prepare_pet_batch(self, batch, mode="PET1"):
        '''
        Prepare for train
        :param batch:
        :return:
        '''

        list_list_txt = [] # [num_text, bs]
        for i in range(1, self.text_ctr):
            list_list_txt.append(batch["input"]["TEXT%d" % i])


        bs = len(batch["input"]["TEXT1"])

        list_input_ids = []
        list_mask_idx = np.ones((bs, self.get_num_lbl_tok())) * self.config.max_text_length

        txt_trim = 1

        for b_idx in range(bs):
            mask_txt_split_tuple = []

            for idx, txt_split in enumerate(self.pattern):
                for i in range(1, self.text_ctr):
                    txt_split = txt_split.replace("[TEXT%d]" % i, list_list_txt[i-1][b_idx])
                txt_split = txt_split.replace("[LBL]", self.tokenizer.mask_token)
                mask_txt_split_tuple.append(txt_split)

            input_ids, mask_idx = tokenize_pet_txt(self.tokenizer, self.config, mask_txt_split_tuple[0], mask_txt_split_tuple[1], mask_txt_split_tuple[2], mask_txt_split_tuple[0], mask_txt_split_tuple[1], mask_txt_split_tuple[2], txt_trim)
            list_input_ids.append(input_ids)
            list_mask_idx[b_idx,:self.get_num_lbl_tok()] = range(mask_idx, mask_idx+self.get_num_lbl_tok())


        return torch.tensor(list_input_ids).to(device), torch.tensor(list_mask_idx).to(device), self.label

    def prepare_pet_mlm_batch(self, batch, mode="PET1"):

        '''
        Prepare for train
        :param batch:
        :return:
        '''

        list_list_txt = [] # [num_text, bs]
        for i in range(1, self.text_ctr):
            list_list_txt.append(batch["input"]["TEXT%d" % i])

        bs = len(batch["input"]["TEXT1"])

        prep_lbl = np.random.randint(self.num_lbl, size=bs)
        tgt = torch.from_numpy(prep_lbl).long() == batch["output"]["lbl"]

        list_orig_input_ids = []
        list_masked_input_ids = []

        txt_trim = 1

        for b_idx, lbl in enumerate(prep_lbl):
            txt_split_tuple = []

            for idx, txt_split in enumerate(self.pattern):
                for i in range(1, self.text_ctr):
                    txt_split = txt_split.replace("[TEXT%d]" % i, list_list_txt[i-1][b_idx])
                txt_split_inp = txt_split.replace("[LBL]", self.label[lbl])
                txt_split_tuple.append(txt_split_inp)

            orig_input_ids, masked_input_ids, mask_idx = tokenize_pet_mlm_txt(self.tokenizer, self.config, txt_split_tuple[0], txt_split_tuple[1], txt_split_tuple[2], txt_trim)
            list_orig_input_ids.append(orig_input_ids)
            list_masked_input_ids.append(masked_input_ids)

        return torch.tensor(list_orig_input_ids).to(device),  torch.tensor(list_masked_input_ids).to(device), prep_lbl, tgt.to(device)

    def prepare_eval_pet_batch(self, batch, mode="PET1"):
        return self.prepare_pet_batch(batch, mode)


    def prepare_batch(self, batch, type):
        '''
        Prepare batch of data for model
        :param batch:
        :param type: pattern to prepare batch with and which mode to use (ex: PET_MLM_PET1)
        :return:
        '''
        # Prepare for PET MLM objective
        if "PET_MLM" in type:
            return self.prepare_pet_mlm_batch(batch, mode=type.replace("PET_MLM_", ""))
        # Prepare for evaluation objective
        elif "EVAL" in type:
            return self.prepare_eval_pet_batch(batch, mode=type.replace("EVAL_", ""))
        # Default is preparing for PET/Decoupled Label objective
        else:
            return self.prepare_pet_batch(batch, mode=type)