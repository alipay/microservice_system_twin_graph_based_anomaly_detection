import hashlib
import json
import logging
import os
import time
import pickle
import random
import numpy as np
import torch
from sklearn.metrics import *
from util.constant import *

def calc_index(predict, actual):
    """
    calculate f1 score by predict and actual.
    """
    if predict.dim() != 2:
        predict = predict.reshape(-1, predict.shape[-1])
    if actual.dim() != 2:
        actual = actual.reshape(-1, actual.shape[-1])

    ap = average_precision_score(actual, predict, average='macro').tolist()
    auc = roc_auc_score(actual, predict, average='macro').tolist()

    if predict.shape[-1] == 2 and actual.shape[-1] == 2:
        actual, predict = torch.argmax(actual, dim=-1), torch.argmax(predict, dim=-1)

    ps = precision_score(actual, predict, average="binary").tolist()
    rs = recall_score(actual, predict, average="binary").tolist()
    effection = f1_score(actual, predict, average="binary", zero_division=1).tolist()

    pred = np.bincount(predict)
    actu = np.bincount(actual)

    if pred.shape[0] == 1:
        information = f'pr:{ps:.4f}  rc:{rs:.4f}  auc:{auc:.4f} ap:{ap:.4f} f1: {effection:.4f} pred_right: {pred[0]} pred_wrong: 0  actu_right: {actu[0]} actu_wrong: {actu[1]}'
    else:
        information = f'pr:{ps:.4f}  rc:{rs:.4f}  auc:{auc:.4f} ap:{ap:.4f} f1: {effection:.4f} pred_right: {pred[0]} pred_wrong:{pred[1]} actu_right: {actu[0]} actu_wrong: {actu[1]}'
    logging.info(information)
    return information, {'pr':ps, 'rc':rs, 'auc':auc, 'ap':ap, 'f1':effection}


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(obj, fw, sort_keys=True, indent=4,
                  separators=(",", ": "), ensure_ascii=False, )


def dump_params(args):
    hash_id = hashlib.md5(str(sorted([(k, v) for k, v in args.items()])).encode("utf-8")).hexdigest()[0:8]
    save_path = os.path.join(args['result_dir'], args['main_model'] + '-' +args['dataset_path'].split('/')[-1] + '-' + hash_id + '-'+ str(int(time.time())))
    os.makedirs(save_path, exist_ok=True)

    log_file = os.path.join(save_path, "running.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,  
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return hash_id, save_path


def read_params(args):
    filename = os.path.join(args['model_path'], "params.json")
    with open(filename) as f:
        dict_json = json.load(fp=f)
   
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    return dict_json


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump_pickle(obj, file_path):
    logging.info("Dumping to {}".format(file_path))
    with open(file_path, "wb") as fw:
        pickle.dump(obj, fw)


def load_pickle(file_path):
    logging.info("Loading from {}".format(file_path))
    with open(file_path, "rb") as fr:
        return pickle.load(fr)
