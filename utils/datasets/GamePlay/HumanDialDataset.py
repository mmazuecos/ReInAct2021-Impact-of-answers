import os
import json
import gzip
import numpy as np
import pandas as pd
import h5py
from PIL import Image
from nltk.tokenize import TweetTokenizer
from torch.utils.data import Dataset
from torchvision import transforms
from utils.image_utils import get_spatial_feat

def create_data_file(model_ans_path, data_dir, data_file, data_args, vocab_file_name, split, model):
    '''Creates the test/val gameplay data given dataset file in *.jsonl.gz format.

    Parameters
    ----------
    data_dir : str
        Directory to read the data and dump the training data created
    data_file : str
        Name of the *.jsonl.gz data file
    data_args : dict
        'successful_only' : bool. Checks what type of games to be included.
        'max_no_objects' : int. Number required for padding of objects in target list for Guesser.
        'max_q_length' : int. Max number of words that QGen can use to ask next question
        'max_src_length' : int. Max number of words that can be present in the dialogue history
        'max_no_qs' : int. Max number of questions that a gamme can have to be included in the data
        'data_paths' : str?. Added by ravi for different file name than default. More details to be added by ravi.
    vocab_file_name : str
        vocabulary file name. This file should have 'word2i' and 'i2word'
    split : str
        Split of the data file
    '''
    path = os.path.join(data_dir, data_file)
    successful_only = data_args['successful_only']
    max_src_length = data_args['max_src_length']

    tknzr = TweetTokenizer(preserve_case=False)
    tmp_key = split + '_process_file'
    if tmp_key in data_args['data_paths']:
        data_file_name = data_args['data_paths'][tmp_key]
    else:
        if successful_only:
            data_file_name = 'human_'+model+'_'+split+'_successful_gameplay_data.json'
        else:
            data_file_name = 'human_'+model+'_'+split+'_all_gameplay_data.json'

    print('Creating new ' + data_file_name + ' file.')

    category_pad_token = 0  # TODO Add this to config.json
    max_no_objects = data_args['max_no_objects']
    no_spatial_feat = 8  # TODO Add this to config.json

    n2n_data = dict()
    _id = 0

    with open(os.path.join(data_dir, vocab_file_name), 'r') as file:
        word2i = json.load(file)['word2i']

    # Load a csv file with the answers of the model to evaluate
    oracle_models_ans = pd.read_csv(os.path.join(data_dir, model_ans_path))

    start = '<start>'

    with gzip.open(path) as file:
        for json_game in file:
            game = json.loads(json_game.decode('utf-8'))

            if successful_only:
                if not game['status'] == 'success':
                    continue

            game_answers = oracle_models_ans[oracle_models_ans.gid == game['id']]
            game_answers.set_index('pos', inplace=True)

            objects = list()
            object_ids = list()
            spatials = list()
            target = int()
            target_cat = int()
            for i, o in enumerate(game['objects']):
                objects.append(o['category_id'])
                object_ids.append(o['id'])
                spatials.append(
                    get_spatial_feat(
                        bbox=o['bbox'],
                        im_width=game['image']['width'],
                        im_height=game['image']['height']
                    )
                )

                if o['id'] == game['object_id']:
                    target = i
                    target_cat = o['category_id']
                    target_spatials = get_spatial_feat(
                        bbox=o['bbox'],
                        im_width=game['image']['width'],
                        im_height=game['image']['height']
                    )

            # Pad objects, spatials and bounding boxes
            objects.extend([category_pad_token] * (max_no_objects - len(objects)))
            object_ids.extend([0] * (max_no_objects - len(object_ids)))
            spatials.extend([[0] * no_spatial_feat] * (max_no_objects - len(spatials)))

            # Compile all of the history
            src = [word2i[start]]
            for i, qa in enumerate(game['qas']):
                q_tokens = tknzr.tokenize(qa['question'])
                q_token_ids = [word2i[w] if w in word2i else word2i['<unk>'] for w in q_tokens][:max_src_length]
                #print(model)
                #print(game_answers)
                ans = game_answers.loc[i][model]
                src += q_token_ids + [word2i[ans]]
            src_length = len(src)
            src.extend([word2i['<padding>']] * (max_src_length - len(src)))
            src = src[:max_src_length]
            src_q = [word2i[start]]

            #n2n_data[_id]['questions'] = q_list
            #n2n_data[_id]['questions_len'] = q_list_lengths
            n2n_data[_id] = dict()
            n2n_data[_id]['history'] = src
            n2n_data[_id]['history_len'] = src_length
            n2n_data[_id]['src_q'] = src_q
            n2n_data[_id]['objects'] = objects
            n2n_data[_id]['spatials'] = spatials
            n2n_data[_id]['target_obj'] = target
            n2n_data[_id]['target_cat'] = target_cat
            n2n_data[_id]['target_spatials'] = target_spatials
            n2n_data[_id]['game_id'] = str(game['id'])
            n2n_data[_id]['image_file'] = game['image']['file_name']
            n2n_data[_id]['image_url'] = game['image']['flickr_url']
            _id += 1

    n2n_data_path = os.path.join(data_dir, data_file_name)
    with open(n2n_data_path, 'w') as f:
        json.dump(n2n_data, f)

    print('Done')

# ---------------------------------------------------------
# ---------------------------------------------------------
class HumanDialDataset(Dataset):
    """docstring for GameplayN2NResNet."""
    def __init__(self, split, model, model_ans_path, **kwargs):
        super(HumanDialDataset, self).__init__()
        self.data_args = kwargs

        visual_feat_file = os.path.join(self.data_args['data_dir'],self.data_args['data_paths']['ResNet']['image_features'] )
        visual_feat_mapping_file  = os.path.join(self.data_args['data_dir'],self.data_args['data_paths']['ResNet']['img2id'] )
        self.vf = np.asarray(h5py.File(visual_feat_file, 'r')[split+'_img_features'])

        with open(visual_feat_mapping_file, 'r') as file_v:
            self.vf_mapping = json.load(file_v)[split+'2id']

        tmp_key = split + "_process_file"
        if tmp_key in self.data_args['data_paths']:
            data_file_name = self.data_args['data_paths'][tmp_key]
        else:
            if self.data_args['successful_only']:
                data_file_name = 'human_'+model+'_'+split+'_successful_gameplay_data.json'
            else:
                data_file_name = 'human_'+model+'_'+split+'_all_gameplay_data.json'

        if self.data_args['new_data'] or not os.path.isfile(os.path.join(self.data_args['data_dir'], data_file_name)):
            create_data_file(model_ans_path, data_dir=self.data_args['data_dir'], data_file=self.data_args['data_paths'][split], data_args=self.data_args, vocab_file_name=self.data_args['data_paths']['vocab_file'], split=split, model=model)

        with open(os.path.join(self.data_args['data_dir'], data_file_name), 'r') as f:
            print('loading ', data_file_name)
            self.game_data = json.load(f)

    def __len__(self) :
        return len(self.game_data)

    def __getitem__(self, idx):
        if not type(idx) == str:
            idx = str(idx)

        # Load image features
        image_file = self.game_data[idx]['image_file']
        visual_feat_id = self.vf_mapping[image_file]
        visual_feat = self.vf[visual_feat_id]
        ImgFeat = visual_feat

        _data = dict()
        _data['history'] = np.asarray(self.game_data[idx]['history'])
        _data['history_len'] = self.game_data[idx]['history_len']
        _data['src_q'] = np.asarray(self.game_data[idx]['src_q'])
        _data['objects'] = np.asarray(self.game_data[idx]['objects'])
        _data['objects_mask'] = np.asarray(1-np.equal(self.game_data[idx]['objects'], np.zeros(len(self.game_data[idx]['objects']))))
        _data['spatials'] = np.asarray(self.game_data[idx]['spatials'])
        _data['target_obj'] = self.game_data[idx]['target_obj']
        _data['target_cat'] = self.game_data[idx]['target_cat']
        _data['target_spatials'] = np.asarray(self.game_data[idx]['target_spatials'], dtype=np.float32)
        _data['image'] = ImgFeat
        _data['image_file'] = image_file
        _data['game_id'] = self.game_data[idx]['game_id']
        _data['image_url'] = self.game_data[idx]['image_url']

        return _data

if __name__ == '__main__':
    split = 'val'
    data_dir = 'data'
    data_file = 'guesswhat.valid.jsonl.gz'
    vocab_file = 'vocab.json'

    data_args = {
        'max_src_length': 200,
        'max_q_length': 30,
        'max_no_objects': 20,
        'max_no_qs': 10,
        'successful_only': True,
        'data_paths': ''
    }

    create_data_file(model_ans_path, data_dir=data_dir, data_file=data_file, data_args=data_args, vocab_file_name=vocab_file, split=split, model='qcs')
