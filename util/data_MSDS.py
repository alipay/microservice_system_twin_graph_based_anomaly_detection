import logging
import os
import pickle
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from util.constant import *

# get service dependency
def read_graph(data_dir):
    logging.info("read Graph edge data")
    if not os.path.exists(data_dir):
        logging.info("read no graph data")
        return None
    data = pickle.load(open(os.path.join(data_dir, 'trace_path.pkl'), 'rb'))
    return data

class Process:
    def __init__(self, **kwargs):

        self.window = kwargs['window']
        self.step = kwargs['step']
        self.dataset_path = kwargs['dataset_path']
        self.rawdata_path = kwargs["data_path"]

        self.log_len = kwargs['log_len']
        self.metric_len = kwargs['raw_node']
        self.num_node = kwargs['num_nodes']
        self.percent = kwargs['label_percent']
        self.set, self.dataset, self.trace_type = {}, [], []

        if os.path.exists(self.dataset_path):
            self.read_data()
            self.graph = read_graph(data_dir=self.rawdata_path)
        else:
            self.load_raw()
            self.graph = read_graph(data_dir=self.rawdata_path)
            logging.info("Tranform data into timewindows")
            self.dataset = self._transform()
            self.save_data()
            
    # reading multidata
    def load_raw(self):
        if not os.path.exists(self.rawdata_path):
            logging.info("Find no data")
        logging.info("LOADing data ...")
        label_raw = pickle.load(open(os.path.join(self.rawdata_path, 'label.pkl'), 'rb'))
        label = np.eye(2)[label_raw.astype(int)]

        label_mask = label_raw.copy()
        times = np.zeros((self.num_node, 2))  
        for idx in range(label.shape[0]):
            if idx < self.window:
                continue
            times += label[idx]
            mask = times[label[idx] == 1] %10 >= 10*self.percent
            label_mask[idx, mask] = 2
        label_mask = np.eye(3)[label_mask.astype(int)]

        metirc = pd.read_csv(os.path.join(self.rawdata_path, 'metric.csv'), sep=',')
        timestart, timeend = metirc['now'].min(), metirc['now'].max()
        time_list = [item for item in range(int(timestart), int(timeend)+1, 1)]
        time_lack = list(set(time_list).difference(set(metirc['now'].values.tolist())))
        for stamp in time_lack:
            metirc = metirc.append([{'now':stamp}])
        metirc = metirc.sort_values(by='now', ascending=True)
        metirc.fillna(method='ffill', inplace=True)
        name_list = list(filter(lambda x: 'mem' in x, list(metirc.columns)))
        after_name = list(map(lambda x: f'{x.split("_")[0]}_a{x.split("_")[-1]}',name_list))
        name_dict = {name_list[idx]: after_name[idx] for idx, _ in enumerate(name_list)}
        metirc.rename(columns = name_dict,  inplace=True)

        log = pd.read_csv(os.path.join(self.rawdata_path, 'log.csv'), sep=',')
        log = log.sort_values(by='@timestamp', ascending=True)
        log_record = {}
        max_record = np.zeros(self.log_len)
        min_record = np.ones(self.log_len)
        for timestamp, data in log.groupby(['@timestamp']):
            new = np.zeros((len(MSDS_pod), self.log_len))
            for idx, item in data.groupby(['Hostname', 'templateid']):
                if idx[0] not in MSDS_pod:
                    continue
                new[MSDS_pod.index(idx[0]), idx[1] - 1] = item.shape[0]
            log_record[timestamp] = new
            new = new.max(axis=0)
            max_record = np.where(new > max_record, new, max_record)
            min_record = np.where(new < min_record, new, min_record)

        if len(log_record) != (timeend - timestart + 1):
            for item in time_list:
                if item not in log_record:
                    log_record[item] = np.zeros((len(MSDS_pod), self.log_len))
            min_record = np.zeros(self.log_len)
        
        dis = max_record - min_record + 1e-6
        for name, item in log_record.items():
            log_record[name] = (item - min_record) / dis

        trace_raw = pd.read_csv(os.path.join(self.rawdata_path, 'trace.csv'), sep=',')
        trace_raw = trace_raw.sort_values(by='end_time', ascending=True)
        self.trace_type.extend(trace_raw['stats'].unique().tolist())
        trace_a = np.zeros((len(MSDS_pod), len(MSDS_pod), len(self.trace_type), len(time_list)))
        for name, item in trace_raw.groupby(['cmbd_id', 'fatherpod', 'stats', 'end_time']):
            if name[0] not in MSDS_pod or name[1] not in MSDS_pod or name[3] > timeend:
                    continue
            trace_a[MSDS_pod.index(name[0]), MSDS_pod.index(name[1]), self.trace_type.index(name[2]), int(name[3]-timestart)] = item['duration'].sum()
        trace = trace_a.transpose(3, 0, 1, 2) / (trace_a.mean(axis=-1)*10 + 1e-6)

        self.set['metric'] = metirc
        self.set['log'] = log_record
        self.set['trace'] = trace
        self.set['label'] = label
        self.set['mask'] = label_mask

    # read data after dealing
    def read_data(self):
        logging.info("read Tranform data")
        if not os.path.exists(self.dataset_path):
            logging.info("read no data")
            return None, None
        
        dataset = os.listdir(self.dataset_path)
        dataset.sort(key=lambda x: (int(re.split(r"[-_.]", x)[0])))
        
        for file in tqdm(dataset):
            data = pickle.load(open(os.path.join(self.dataset_path, file), 'rb'))
            self.dataset.append(data)

    # saving data
    def save_data(self):
        logging.info("save Tranform data")
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)
        for _, item in tqdm(enumerate(self.dataset)):
            with open(f'{self.dataset_path}/{item["name"]}.pkl', 'wb') as f:
                del item['name']
                pickle.dump(item, f)

    # split to sliding windows
    def _transform(self):
        self.trace_type = list(set(self.trace_type))
        num = 0
        count1, count2, count3 = 0, 0, 0
        data_list = []

        metirc = self.set['metric']
        log = self.set['log']
        trace = self.set['trace']
        label = self.set['label']
        label_mask = self.set['mask']

        starttime, endtime = metirc['now'].min(), metirc['now'].max()

        metirc.sort_index(axis=1, ascending=True, inplace=True)
        
        while starttime + (self.window - 1) * self.step <= endtime:
            record = {}
            if num % 500 == 0:
                logging.info(f"deal ...{num}...trace:{count3}...error see:{count1}...error real:{count2}...{starttime}")
            
            #metric
            select_metirc = metirc[(metirc['now'] >= starttime) & (metirc['now'] <= starttime + (self.window - 1) * self.step)]
            select_metirc.set_index('now', inplace=True)
            select_metirc = select_metirc.values
            select_metirc = select_metirc.reshape(self.window, 5, -1)
            assert select_metirc.shape == (self.window, len(MSDS_pod), 5), f"Worng kpi"
            record['data_node'] = select_metirc[:, :, :self.metric_len]

            # log
            log_record = np.stack([log[time] for time in range(int(starttime), int(starttime + (self.window - 1) * self.step + 1), self.step)], axis=0)
            record['data_log'] = np.nan_to_num(log_record)
            assert log_record.shape == (self.window, len(MSDS_pod), self.log_len), f"Worng log"

            #label
            select_label = label[num + self.window - 1, :]
            select_mask = label_mask[num + self.window - 1, :]

            record['groundtruth_cls'] = select_mask
            record['groundtruth_real'] = select_label
            count1 += 1 if record['groundtruth_cls'].sum(axis=0)[1] > 0 else 0
            count2 += 1 if record['groundtruth_real'].sum(axis=0)[1] > 0 else 0
            assert record['groundtruth_cls'].shape == (len(MSDS_pod), 3), f"Worng label"    

            # trace
            select_trace = trace[num : num + self.window]
            count3 += 1 if select_trace.sum() > 0 else 0
            assert select_trace.shape == (self.window, len(MSDS_pod), len(MSDS_pod), len(self.trace_type)), f"Worng Trace"
            record['data_edge'] = select_trace
            record['name'] = f'{num}'
            num += 1
            data_list.append(record)
            starttime += self.step
            del record
        logging.info(f"deal ...{num}...error see:{count1}...error real:{count2}...")
        return data_list
