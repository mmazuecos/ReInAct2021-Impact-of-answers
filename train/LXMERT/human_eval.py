import argparse
import json
import multiprocessing
import sys
from time import time
from time import sleep

import numpy as np
import os
import torch.nn as nn
import tqdm
import transformers
from shutil import copy2
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import torch

from models.CNN import ResNet
from models.Ensemble import Ensemble
#from models.Oracle import Oracle
from train.GamePlay.parser import preprocess_config
from utils.datasets.GamePlay.GamePlayLXMERTDataset import GamePlayLXMERTDataset
from utils.datasets.GamePlay.GameplayN2NLXMERTOracleDataset import GameplayN2NLXMERTOracleDataset
from utils.eval import calculate_accuracy
from utils.gameplayutils import *
from utils.model_loading import load_model

from lxmert_oracle.models import LXMERTOracle

# TODO Make this capitalised everywhere to inform it is a global variable
use_cuda = torch.cuda.is_available()

#TODO: Move this code from the train folder

def tensor2text(newq, i2word):
    """
    given a dialoage tensor (BxL) with word token ids, returns the words
    """
    if isinstance(newq, Variable):
        newq = newq.data

    batch_dial = list()

    for bid in range(newq.size(0)):
        dial = str()
        for i in newq[bid]:
            if i2word[str(i.item())] == "<padding>":
                break
            else:
                dial += i2word[str(i.item())] + ' '

        dial += '?'
        batch_dial.append(dial)

    return batch_dial

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/GamePlay/config.json", help=' General config file')
    parser.add_argument("-ens_config", type=str, default="config/GamePlay/ensemble.json", help=' Ensemble config file')
    parser.add_argument("-or_config", type=str, default="config/GamePlay/oracle.json", help=' Oracle config file')
    parser.add_argument("-exp_name", type=str, help='Experiment Name')
    parser.add_argument("-my_cpu", action='store_true', help='To select number of workers for dataloader. CAUTION: If using your own system then make this True')
    parser.add_argument("-breaking", action='store_true',
                        help='To Break training after 5 batch, for code testing purpose')
    parser.add_argument("-resnet", action='store_true', help='This flag will cause the program to use the image features from the ResNet forward pass instead of the precomputed ones.')
    parser.add_argument("-dataparallel", action='store_true', help='This for model files which were saved with Dataparallel')
    parser.add_argument("-log_enchidden", action='store_true', help='This flag saves the encoder hidden state. WARNING!!! This might cause the resulting json file to blow up!')
    parser.add_argument("-max_epochs", type=int, required=True)
    parser.add_argument("-min_epochs", type=int, required=True)


    # --------Arguments from config.json that can be overridden here. Similar changes have to be made in the util file and not here--------------------
    parser.add_argument("-batch_size", type=int, help='Batch size for the gameplay')
    parser.add_argument("-load_bin_path", type=str, help='Bin file path for the saved model. If this is not given then one provided in ensemble.json will be taken ')
    parser.add_argument("-oracle_path", type=str)

    args = parser.parse_args()
    print(args.exp_name)
    use_dataparallel = args.dataparallel
    breaking = args.breaking

    # Load the Arguments and Hyperparamters
    ensemble_args, dataset_args, optimizer_args, exp_config, oracle_args, word2i, i2word, catid2str = preprocess_config(args)

    pad_token= word2i['<padding>']

    torch.manual_seed(exp_config['seed'])
    if use_cuda:
        torch.cuda.manual_seed_all(exp_config['seed'])

    if exp_config['logging']:
        log_dir = exp_config['logdir']+str(args.exp_name)+exp_config['ts']+'/'
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        copy2(args.config, log_dir)
        copy2(args.ens_config, log_dir)
        copy2(args.or_config, log_dir)
        with open(log_dir+'args.txt', 'w') as f:
            f.write(str(vars(args))) # converting args.namespace to dict

    # --- LOAD LXMERT ORACLE ---
    oracle = LXMERTOracle()
    oracle.cuda()
    checkpoint = torch.load(
        args.oracle_path,
        map_location=lambda storage, loc: storage
    )
    oracle.load_state_dict(checkpoint["model_state_dict"])
    oracle.eval()
    

    # --- LOAD LXMERT TOKENIZER ---
    tokenizer = transformers.LxmertTokenizer.from_pretrained(
                'unc-nlp/lxmert-base-uncased'
            )


    # LOAD ENSEMBLE MODEL
    model = Ensemble(**ensemble_args)
    model = load_model(model, model_filename, use_dataparallel=use_dataparallel)

    # TODO Custom Dataloader and dataset for this task
    dataset = 
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=optimizer_args['batch_size'],
        shuffle=False, # This is made False to check model performance at batch level.
        num_workers= 0 if sys.gettrace() else multiprocessing.cpu_count()//2,
        pin_memory= use_cuda,
        drop_last=False)

    # Logging data
    total_no_batches = len(dataloader)
    accuracy = list()
    decider_perc = list()
    start = time()
    # Start to go through the games
    for i_batch, sample in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):

        # Get Batch
        for k, v in sample.items():
            if torch.is_tensor(v):
                sample[k] = to_var(v, True)

        # Get visual features
        avg_img_features = sample['image']

        # Create the history
        for q in sample['questions']:
            # --- LXMERT TOKENIZE NEWQ ---
            langdata = tokenizer(
                q,
                return_tensors="pt",
                padding="max_length",
                return_token_type_ids=False,
                return_attention_mask=True,
                add_special_tokens=True,
                truncation=True,
                max_length=20
            )
        
        # --- LXMERT TOKENIZE NEWQ ---
        langdata['input_ids'] = langdata['input_ids'].cuda()
        langdata['attention_mask'] = langdata['attention_mask'].cuda()

        visdata_keys = [k for k in sample.keys() if k.startswith('lxmert')]
        visdata = { _key[7:]: sample[_key]
                                for _key in visdata_keys
                               }

        # Get history from preprocessed data
        history = to_var(torch.LongTensor(batch_size, 200).fill_(pad_token))
        history[:,0] = sample['history'].squeeze()
        history_len = sample['history_len']

        # Generate the guesser_probs
        if use_dataparallel and use_cuda:
            encoder_hidden = model.module.encoder(history=history, visual_features=avg_img_features, history_len=history_len)
            guesser_logits = model.module.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress= False)
        else:
            encoder_hidden = model.encoder(history=history, visual_features=avg_img_features, history_len=history_len)
            guesser_logits = model.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress= False)

        tmp_guesser_prob = softmax(tmp_guesser_logits * sample['objects_mask'].float())
        batch_accuracy = calculate_accuracy(softmax(guesser_logits*sample['objects_mask'].float()), sample['target_obj'])
        accuracy.append(batch_accuracy)
        decider_perc.append((torch.sum(decisions.data)/decisions.size(0)).item())
