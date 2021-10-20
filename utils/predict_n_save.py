import numpy as np
import pandas as pd
import os
import argparse

import sys

import torch
import jsonlines
import gzip
from tqdm import tqdm

from models.Oracle import Oracle

from utils.config import load_config
from utils.model_loading import load_model
from utils.datasets.Oracle.OracleDataset import OracleDataset

from gw_utils import *

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda') if USE_CUDA else torch.device('cpu')

def get_raw_qas(game):
    """
    Gets the list of questions and their answers from a played game. These
    played games are taken from the human-human GuessWhat!? corpus.

    Parameters
    ----------
    pgame: dict
        data of a played game from the human-human corpus

    Returns
    -------
    out: list
        a list of tuples (question, answer) for each question in game
    """                                                                                                                                                                         
    qas = []
    for qa in game['qas']:
        qfixed = qa['question'][:-1]+' ?'
        # Get rid of the '?' of the last word
        qas.append((qfixed, qa['answer'].lower()))
    return qas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-breaking", action='store_true', help='Run with just one sample for test purpose')
    parser.add_argument("-split", type=str,default='test', help='Partition of the dataset to  use. Options are "train", "val" or "test". Default: val.')
    parser.add_argument("-config", type=str, default='config/Oracle/config.json', help='Path to config file.')
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-out_fname", type=str, default='out.csv.gz', help='Name of the generated file.')
    parser.add_argument("-bin_path", type=str, required=True, help='Path to the Oracle bin model.')
    parser.add_argument("-model_name", type=str, default='model',help='Model name to store in output file')
    parser.add_argument("-img_feat", type=str, default="rss", help='Select "vgg" or "res" as image features')
    parser.add_argument("-new_data", action='store_true', default=False, help='Recompute the Oracle dataset.')

    
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Experiment Settings
    exp_config = config['exp_config']
    exp_config['img_feat'] = args.img_feat.lower()
    exp_config['use_cuda'] = USE_CUDA

    # Hyperparamters
    data_paths          = config['data_paths']
    optimizer_config    = config['optimizer']
    embedding_config    = config['embeddings']
    lstm_config         = config['lstm']
    mlp_config          = config['mlp']
    dataset_config      = config['dataset']
    inputs_config       = config['inputs']

    # Load vocabulary
    with open(os.path.join(args.data_dir, data_paths['vocab_file'])) as file:
        vocab = json.load(file)
    word2i = vocab['word2i']
    i2word = vocab['i2word']
    vocab_size = len(word2i)

    # Transform tokens to actual words
    tok2ans = {0:'<no>', 1: '<yes>', 2: '<n/a>'}

    # Init model and load weights
    model = Oracle(
        no_words            = vocab_size,
        no_words_feat       = embedding_config['no_words_feat'],
        no_categories       = embedding_config['no_categories'],
        no_category_feat    = embedding_config['no_category_feat'],
        no_hidden_encoder   = lstm_config['no_hidden_encoder'],
        mlp_layer_sizes     = mlp_config['layer_sizes'],
        no_visual_feat      = inputs_config['no_visual_feat'],
        no_crop_feat        = inputs_config['no_crop_feat'],
        dropout             = lstm_config['dropout'],
        inputs_config       = inputs_config,
        scale_visual_to     = inputs_config['scale_visual_to']
    )
    model = load_model(model, args.bin_path,
                       use_dataparallel=False)
    model.to(device)
	# Set it to eval so no backprop computing
    model.eval()

    # Load dataset
    dataset = OracleDataset(
        data_dir            = args.data_dir,
        data_file           = data_paths[args.split+'_file'],
        split               = args.split,
        visual_feat_file    = data_paths[args.img_feat]['image_features'],
        visual_feat_mapping_file = data_paths[exp_config['img_feat']]['img2id'],
        visual_feat_crop_file = data_paths[args.img_feat]['crop_features'],
        visual_feat_crop_mapping_file = data_paths[exp_config['img_feat']]['crop2id'],
        max_src_length      = dataset_config['max_src_length'],
        hdf5_visual_feat    = args.split+'_img_features',
        hdf5_crop_feat      = 'objects_features',
        history             = dataset_config['history'],
        new_oracle_data     = args.new_data,
        successful_only     = False,
        load_crops          = inputs_config['crop'],
        negative            = dataset_config['negative'],
        supercats           = dataset_config['supercats'],
        second              = dataset_config['second'],
    )

    # Load game ids
    rawdata_path = os.path.join(args.data_dir, data_paths[args.split+'_file']) 
    rawdata = jsonlines.Reader(gzip.open(rawdata_path))

    # Data to return:
    # key: {'qas': a list of questions
    #       'state': either success, failure or incomplete}
    mkname = args.model_name + '_ans'
    data = {'question_id':[],
            mkname: [],
            'hist': []}

    # DataLoader that yields the needed data
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0 if sys.gettrace() else 4,
        pin_memory=USE_CUDA
    )

    # Predict answer for each question
    prev_game_id = '-1'
    ingame_count = 0
    for sample in tqdm(dataloader):

        # For test purposes
        #if args.breaking and j > 5:
        #    break
        # Get question
        questions, answers, crop_features, visual_features, spatials, focus, obj_categories, lengths, game_id, hist, qid = \
            sample['question'], sample['answer'], sample['crop_features'], sample['img_features'], sample['spatial'], sample['focus'], sample['obj_cat'], sample['length'], sample['game_id'],sample['hist'], sample['question_id']

        # This would only work with shuffle=False in dataloader
        if game_id[0] != prev_game_id:
            ingame_count = 0
            prev_game_id = game_id[0]
        else:
            ingame_count += 1

        # Forward pass
        pred_answer = model(questions.to(device),
            obj_categories.to(device),
            spatials.to(device),
            focus.to(device),
            crop_features.to(device),
            visual_features.to(device),
            lengths.to(device)
        )

        pred = pred_answer.argmax(dim=1).cpu().tolist()
        data['question_id'] += qid.cpu().tolist()
        data[mkname] += [tok2ans[p] for p in pred]
        # This one is a list of strings, those are not parallelized by pytorch
        data['hist'] += hist
        #print(qid)
        assert(len(data['question_id']) == len(data[mkname]))
        assert(len(data['question_id']) == len(data['hist']))
        assert(len(data['hist']) == len(data[mkname]))

	# Save file
    df = pd.DataFrame(data) 
    df.to_csv(args.out_fname, index=False, compression='gzip')
