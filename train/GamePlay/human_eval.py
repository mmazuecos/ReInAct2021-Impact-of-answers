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
from models.Oracle import Oracle
from train.GamePlay.parser import preprocess_config
from utils.datasets.GamePlay.HumanDialDataset import HumanDialDataset
from utils.eval import calculate_accuracy
from utils.gameplayutils import *
from utils.model_loading import load_model

# TODO Make this capitalised everywhere to inform it is a global variable
use_cuda = torch.cuda.is_available()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/GamePlay/config.json", help=' General config file')
    parser.add_argument("-load_bin_path", type=str, help='Bin file path for the saved model. If this is not given then one provided in ensemble.json will be taken ')
    parser.add_argument("-ens_config", type=str, default="config/GamePlay/ensemble.json", help=' Ensemble config file')
    parser.add_argument("-or_config", type=str, default="config/GamePlay/oracle.json", help=' Oracle config file')
    parser.add_argument("-exp_name", type=str, help='Experiment Name')
    parser.add_argument("-my_cpu", action='store_true', help='To select number of workers for dataloader. CAUTION: If using your own system then make this True')
    parser.add_argument("-resnet", action='store_true', help='This flag will cause the program to use the image features from the ResNet forward pass instead of the precomputed ones.')
    parser.add_argument("-use_model", type=str)
    parser.add_argument("-breaking", action='store_true',
                        help='To Break training after 5 batch, for code testing purpose')
    parser.add_argument("-dataparallel", action='store_true', help='This for model files which were saved with Dataparallel')
    parser.add_argument("-log_enchidden", action='store_true', help='This flag saves the encoder hidden state. WARNING!!! This might cause the resulting json file to blow up!')
    parser.add_argument("-model_ans_path", type=str, required=True, default='model_ans.csv.gz')

    # --------Arguments from config.json that can be overridden here. Similar changes have to be made in the util file and not here--------------------
    parser.add_argument("-batch_size", type=int, help='Batch size for the gameplay')
    parser.add_argument("-oracle_path", type=str)
    parser.add_argument("-model_filename", type=str)
    parser.add_argument("-split", default='test', type=str)

    args = parser.parse_args()
    print(args.exp_name)
    use_dataparallel = args.dataparallel

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
 
    # LOAD ENSEMBLE MODEL
    model = Ensemble(**ensemble_args)
    model = load_model(model, args.model_filename, use_dataparallel=use_dataparallel)

    # TODO Custom Dataloader and dataset for this task
    dataset = HumanDialDataset(split=args.split,
                               model=args.use_model,
                               model_ans_path=args.model_ans_path,
                               **dataset_args) 

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

    softmax = nn.Softmax(dim=-1)

    #model_name = 'devries' if args.use_model == 'qcs' else 'lxmert'
    print('working with: ', args.use_model) 
    # ---------------------------
    guesser_probs_log = {}
    status = {True: 'success', False: 'failure'}
    data = {'gid': [], args.use_model+'_status': []}
    # Start to go through the games
    for i_batch, sample in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):

        # ---------------------------
        # Get Batch
        for k, v in sample.items():
            if torch.is_tensor(v):
                sample[k] = to_var(v, True)

        # ---------------------------
        # Get visual features
        avg_img_features = sample['image']

        # ---------------------------
        # Get batch size
        batch_size = avg_img_features.shape[0]

        # ---------------------------
        # Get history from preprocessed data
        history = to_var(torch.LongTensor(batch_size, 200).fill_(pad_token))
        history = sample['history'].squeeze()
        history_len = sample['history_len']

        # ---------------------------
        # Generate the guesser_probs
        if use_dataparallel and use_cuda:
            encoder_hidden = model.module.encoder(history=history, visual_features=avg_img_features, history_len=history_len)
            guesser_logits = model.module.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress=False)
        else:
            encoder_hidden = model.encoder(history=history, visual_features=avg_img_features, history_len=history_len)
            guesser_logits = model.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress=False)

        # ---------------------------
        # Compute accuracy
        guesser_prob = softmax(guesser_logits * sample['objects_mask'].float())
        batch_accuracy, results = calculate_accuracy(softmax(guesser_logits*sample['objects_mask'].float()), sample['target_obj'])
        accuracy.append(batch_accuracy)
        data['gid'] += sample['game_id']
        data[args.use_model+'_status'] += [status[i] for i in results.detach().cpu().numpy()]

        # ---------------------------
        # Logging guesser_prob
        log_probs = guesser_prob.cpu().detach().numpy().tolist()
        for j, gid in enumerate(sample['game_id']):
            guesser_probs_log[gid] = log_probs[j]


    # Print results
    current_acc = np.mean(accuracy)
    print(args.split + ' accuracy ' + str(current_acc))

    # Save log
    with open(args.exp_name + '_log.json', 'w') as fl:
        json.dump(data, fl)
