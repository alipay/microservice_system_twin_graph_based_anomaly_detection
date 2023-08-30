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
        self.dataset = []
        self.log_len = kwargs['log_len']
        self.trace_type = []
        self.dataset_path = kwargs['dataset_path']
        self.rawdata_path = kwargs["data_path"]
        self.set = {}

        if os.path.exists(self.dataset_path):
            self.read_data()
            self.graph = read_graph(data_dir=self.rawdata_path)
        else:
            self.load_raw()
            self.graph = read_graph(data_dir=self.rawdata_path)
            logging.info("Tranform data into timewindows")
            self.dataset = self._transform()
            self.save_data()
            
    # 读取多源数据
    def load_raw(self):
        if not os.path.exists(self.rawdata_path):
            logging.info("Find no data")
        logging.info("LOADing data ...")
        label = pickle.load(open(os.path.join(self.rawdata_path, 'label.pkl'), 'rb'))
        label_mask = np.ones(label.shape[0]) *2 
        need, times = 0, 0  #need anomaly; times normal
        for idx, item in enumerate(label.sum(axis=1)):
            if idx < 10:
                continue
            if item > 0:
                label_mask[idx] = 1 if times < 10 else 2   #abnormal
                times += 1
            else:
                count = label[idx-self.window + 1:idx +1].sum()
                if count == 0 and need < 10:   #normal
                    label_mask[idx] = 0
                need += 1
                    
            if times == 20:
                times = 0
            if need == 20:
                need = 0            
        
        metirc = pd.read_csv(os.path.join(self.rawdata_path, 'metric.csv'), sep=',')
        timestart, timeend = metirc['now'].min(), metirc['now'].max()
        time_list = [item for item in range(int(timestart), int(timeend)+1, 1)]
        time_lack = list(set(time_list).difference(set(metirc['now'].values.tolist())))
        for stamp in time_lack:
            metirc = metirc.append([{'now':stamp}])
        metirc = metirc.sort_values(by='now', ascending=True)
        metirc.fillna(method='ffill', inplace=True)
        name_list = list(filter(lambda x: 'mem' in x, list(metirc.columns)))
        after_name = list(map(lambda x: f'{x.split("_")[0]}_a{x.split("_")[-1]})',name_list))
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

        trace = pd.read_csv(os.path.join(self.rawdata_path, 'trace.csv'), sep=',')
        trace = trace.sort_values(by='end_time', ascending=True)
        self.trace_type.extend(trace['stats'].unique().tolist())

        self.set['metric'] = metirc
        self.set['log'] = log_record
        self.set['trace'] = trace
        self.set['label'] = label
        self.set['mask'] = label_mask

    # 读取已经存在的数据
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

    # 保存已经存在的数据
    def save_data(self):
        logging.info("save Tranform data")
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)
        for _, item in tqdm(enumerate(self.dataset)):
            with open(f'{self.dataset_path}/{item["name"]}.pkl', 'wb') as f:
                del item['name']
                pickle.dump(item, f)

    # 把读取的数据依据滑动窗口切分
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
        
        # Sliding window
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
            record['data_node'] = select_metirc

            # log
            log_record = np.stack([log[time] for time in range(int(starttime), int(starttime + (self.window - 1) * self.step + 1), self.step)], axis=0)
            record['data_log'] = np.nan_to_num(log_record)
            assert log_record.shape == (self.window, len(MSDS_pod), self.log_len), f"Worng log"

            #label
            select_label_pod = label[num + self.window - 1, :]
            select_mask = label_mask[num + self.window - 1]
            if select_mask == 2:
                result = np.ones_like(select_label_pod) * 2   #unknown
            else:
                result = select_label_pod

            record['groundtruth_cls'] = np.eye(3)[result.astype(int)]
            record['groundtruth_real'] = np.eye(2)[select_label_pod.astype(int)]
            count1 += 1 if record['groundtruth_cls'].sum(axis=0)[1] > 0 else 0
            count2 += 1 if record['groundtruth_real'].sum(axis=0)[1] > 0 else 0
            assert record['groundtruth_cls'].shape == (len(MSDS_pod), 3), f"Worng label"    

            # trace
            select_trace = trace[(trace['end_time'] >= starttime) & ( trace['end_time'] <= starttime + (self.window - 1) * self.step)]
            A = np.zeros((len(MSDS_pod), len(MSDS_pod), len(self.trace_type),self.window))
            for name, item in select_trace.groupby(['cmbd_id', 'fatherpod', 'stats']):
                if name[0] not in MSDS_pod or name[1] not in MSDS_pod:
                    continue
                time = {starttime + k * self.step: 0 for k in range(self.window)}
                for timestamp, record_data in item.groupby('end_time'):
                    time[timestamp] = record_data['duration'].sum()

                A[MSDS_pod.index(name[1])][MSDS_pod.index(name[0])][self.trace_type.index(name[2])] = np.array(list(time.values()))
                A[MSDS_pod.index(name[0])][MSDS_pod.index(name[1])][self.trace_type.index(name[2])] = np.array(list(time.values()))
            
            A = A.transpose(3, 0, 1, 2)
            count3 += 1 if A.sum() > 0 else 0
            assert A.shape == (self.window, len(MSDS_pod), len(MSDS_pod), len(self.trace_type)), f"Worng Trace"
            record['data_edge'] = A
            record['name'] = f'{num}'
            num += 1
            data_list.append(record)
            starttime += self.step
            del record
        logging.info(f"deal ...{num}...error see:{count1}...error real:{count2}...")
        return data_list
