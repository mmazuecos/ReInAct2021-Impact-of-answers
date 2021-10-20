import json

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.datasets.GamePlay.prepro_lxmert import create_data_file
from utils.lxmert_spatial_utils import xyxy2xywh, xywh2xyxy

import transformers

_ANSWER_TO_LABEL_MAP = {"No": 0, "Yes": 1, "N/A": 2}


class COCODataProvider(object):
    def __init__(self, root, num_objects=36):
        super(COCODataProvider, self).__init__()
        self.root = os.path.abspath(root)
        self.num_objects = int(num_objects)

    def get_data(self, imkey):
        coco_set = imkey.split("_")[1]
        imfile = os.path.join(
            self.root, coco_set, os.path.splitext(imkey)[0] + ".npy"
        )
        data = np.load(imfile, allow_pickle=True).item()
        boxes, features = data["boxes"], data["features"]

        if len(boxes) == 0:
            boxes = np.zeros((self.num_objects, 4), dtype=np.float32)
            features = np.zeros((self.num_objects, 2048), dtype=np.float32)
        elif boxes.shape[0] < self.num_objects:
            n = self.num_objects - boxes.shape[0]
            boxes = np.vstack([boxes, np.zeros((n, 4), dtype=np.float32)])
            features = np.vstack([features, np.zeros((n, 2048), dtype=np.float32)])
        else:
            boxes = boxes[:self.num_objects]
            features = features[:self.num_objects]

        return boxes, features

class COCODataProviderSingleFile(object):
    def __init__(self, root, num_objects=36):
        self.boxes = np.load(
            os.path.join(root, "mscoco_bottomup_boxes.npy")
        )  # in (x1, y1, x2, y2) format

        self.features = np.load(
            os.path.join(root, "mscoco_bottomup_features.npy"),
            mmap_mode="c"
        )

        self.info = json.load(open(
            os.path.join(root, "mscoco_bottomup_info.json")
        ))

        assert num_objects == 36

    def get_data(self, imkey):
        if imkey.endswith(".jpg"):
            imkey = os.path.splitext(imkey)[0]
        pos = self.info["image_id2image_pos"][imkey]
        features = self.features[pos]
        boxes = xyxy2xywh(self.boxes[pos])  # (x,y,w,h) by default
        return boxes, features


class GameplayN2NLXMERTOracleDataset(Dataset):
    """docstring for GameplayN2NResNet."""

    def __init__(self, split, **kwargs):
        super(GameplayN2NLXMERTOracleDataset, self).__init__()
        self.data_args = kwargs

        print('kwargs ', kwargs) 
        print('type(kwargs)', type(kwargs))
        # NEED TO BE PASSED BY ARGUMENT
        coco_data_root = "./lxmert_oracle/fasterrcnn/mscoco_num-objects_36/"
        if os.path.exists(os.path.join("./lxmert_oracle/", "mscoco_bottomup_boxes.npy")):
            self.coco_data_provider = COCODataProviderSingleFile(coco_data_root)
        else:
            self.coco_data_provider = COCODataProvider(coco_data_root)

        oracle_targets_file = "./lxmert_oracle/resnet152/guesswhat_test_oracle.npy"
        self.targets = np.load(oracle_targets_file, allow_pickle=True).item()

        tmp_key = split + "_process_file"

        self.img_dir = os.path.join(self.data_args['data_paths']['image_path'], split)

        if tmp_key in self.data_args['data_paths']:
            data_file_name = self.data_args['data_paths'][tmp_key]
        else:
            if self.data_args['successful_only']:
                data_file_name = 'n2nlxmert_' + split + '_successful_gameplay_data.json'
            else:
                data_file_name = 'n2nlxmert_' + split + '_all_gameplay_data.json'

        if self.data_args['new_data'] or not os.path.isfile(os.path.join(self.data_args['data_dir'], data_file_name)):
            create_data_file(
                data_dir=self.data_args['data_dir'],
                data_file=self.data_args['data_paths'][split],
                data_args=self.data_args,
                vocab_file_name=self.data_args['data_paths']['vocab_file'],
                split=split
            )

        with open(os.path.join(self.data_args['data_dir'], data_file_name), 'r') as f:
            self.game_data = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.game_data)

    def __getitem__(self, idx):
        if not type(idx) == str:
            idx = str(idx)

        # Load image features
        image_file = self.game_data[idx]['image_file']
        tmp_img_path = os.path.join(self.img_dir, image_file)
        if os.path.isfile(tmp_img_path):
            img_path = tmp_img_path
        else:
            # Taking care if image is stored as in MS-COCO directory structure
            tmp_img_path = os.path.join(self.data_args['data_paths']['image_path'], 'train2014', image_file)
            if os.path.isfile(tmp_img_path):
                img_path = tmp_img_path
            else:
                tmp_img_path = os.path.join(self.data_args['data_paths']['image_path'], 'val2014', image_file)
                if os.path.isfile(tmp_img_path):
                    img_path = tmp_img_path
                else:
                    print('Something wrong with image path')

        # -------------------
        # --- LXMERT data ---
        # -------------------
        # read detections
        boxes, features = self.coco_data_provider.get_data(image_file)
        #boxes = torch.from_numpy(boxes)

        boxes = xywh2xyxy(boxes)

        #features = torch.from_numpy(features)

        # get target data
        target = self.targets[self.game_data[idx]['game_id']]

        target_box = target["boxes"].copy()
        #target_box = torch.from_numpy(target_box)

        target_box = xywh2xyxy(target_box)

        target_feature = target["features"].copy()
        #target_feature = torch.from_numpy(target_feature)
        #valid_boxes = (boxes.abs().sum(-1) > 1e-4).float()
        valid_boxes = (abs(boxes).sum(-1) > 1e-4).float()
        width = self.game_data[idx]["img_width"]
        height = self.game_data[idx]["img_height"]

        boxes[:, (0, 2)] /= width
        boxes[:, (1, 3)] /= height

        target_box[0, (0, 2)] /= width
        target_box[0, (1, 3)] /= height

        visdata = {
            "visual_feats": features,
            "visual_pos": boxes,
            "visual_attention_mask": valid_boxes,
            "target_feat": target_feature,
            "target_pos": target_box
        }
        # -------------------
        # ---/LXMERT data ---
        # -------------------

        ImgTensor = self.transform(Image.open(img_path).convert('RGB'))
        _data = dict()
        _data['history'] = np.asarray(self.game_data[idx]['history'])
        _data['history_len'] = self.game_data[idx]['history_len']
        _data['src_q'] = np.asarray(self.game_data[idx]['src_q'])
        _data['objects'] = np.asarray(self.game_data[idx]['objects'])
        _data['objects_mask'] = np.asarray(1 - np.equal(self.game_data[idx]['objects'], np.zeros(len(self.game_data[idx]['objects']))))
        _data['spatials'] = np.asarray(self.game_data[idx]['spatials'])
        _data['target_obj'] = self.game_data[idx]['target_obj']
        _data['target_cat'] = self.game_data[idx]['target_cat']
        _data['target_spatials'] = np.asarray(self.game_data[idx]['target_spatials'], dtype=np.float32)
        _data['image'] = ImgTensor
        _data['image_file'] = image_file
        _data['game_id'] = self.game_data[idx]['game_id']
        _data['image_url'] = self.game_data[idx]['image_url']
        _data['lxmert_visdata'] = visdata

        return _data
