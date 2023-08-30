import time
import os
import sys
import pickle
import json
import jsonpickle
import logging
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from cachetools import LRUCache
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

start = '2019-11-25 15:12:13'
end = '2019-11-25 18:18:13'
start_dealtime  = time.mktime(time.strptime(start, '%Y-%m-%d %H:%M:%S'))
end_dealtime  = time.mktime(time.strptime(end, '%Y-%m-%d %H:%M:%S'))
MSDS_pod = ['wally113','wally117','wally122','wally123','wally124']
raw_path = './data/MSDS/concurrent_data'
save_path = './data/MSDS-pre'
os.makedirs(save_path, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

def stamptotime(num):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(num))


def save_state(template_miner, filename):
    f = open(filename, 'wb')
    state = jsonpickle.dumps(template_miner.drain, keys=True).encode('utf-8')
    f.write(state)


def save_template(template_miner, filename):
    with open(filename, 'w') as f:
        for lines in template_miner.drain.clusters:
            f.write(str(lines.cluster_id) + "_" + lines.get_template() + '\n')


def createparser(configfile, statefile=" "):
    if not os.path.isfile(configfile) and not os.path.isfile(statefile):
        logger.info(f"There are no config file")
        return False

    if os.path.isfile(statefile):
        logger.info(f"load state ....")
        config = TemplateMinerConfig()
        config.load(configfile)
        config.profiling_enabled = True
        template_miner = TemplateMiner(config=config)
        f = open(statefile)
        state = f.read()
        loaded_drain = jsonpickle.loads(state, keys=True)
        if len(loaded_drain.id_to_cluster) > 0 and isinstance(next(iter(loaded_drain.id_to_cluster.keys())), str):
            loaded_drain.id_to_cluster = {int(k): v for k, v in list(loaded_drain.id_to_cluster.items())}
            if template_miner.config.drain_max_clusters:
                cache = LRUCache(maxsize=template_miner.config.drain_max_clusters)
                cache.update(loaded_drain.id_to_cluster)
                loaded_drain.id_to_cluster = cache

        template_miner.drain.id_to_cluster = loaded_drain.id_to_cluster
        template_miner.drain.clusters_counter = loaded_drain.clusters_counter
        template_miner.drain.root_node = loaded_drain.root_node
        return template_miner

    # logger.info(f"create state ....")
    config = TemplateMinerConfig()
    config.load(configfile)
    config.profiling_enabled = True
    template_miner = TemplateMiner(config=config)
    return template_miner


def logparse(miner, rawlog):
    start_time = time.time()
    templateid_list = []

    for line in tqdm(rawlog):
        if type(line) != str:
            line = str(line)
        line = line.rstrip()
        result = miner.add_log_message(line)
        templateid_list.append(result["cluster_id"])

    time_took = time.time() - start_time
    rate = rawlog.shape[0] / time_took
    logger.info(
        f"--- Done processing file in {time_took:.2f} sec. Total of {rawlog.shape[0]} lines, rate {rate:.1f} lines/sec, "
        f"{len(miner.drain.clusters)} clusters")
    return templateid_list

def deal_ground(datapath):
    for name in os.listdir(datapath):
        namel =  re.split(r'[_.]', name)
        if namel[-1] != 'json':
            continue

        with open(os.path.join(datapath, name)) as f:
            data = json.load(f)  # Output: dict

        data = data['tasks'][0]["subtasks"][0]['workloads'][0]['data']
        result = []
        for item in data:
            timesstamp = item['timestamp']
            error = 1 if item['error'] else 0
            duration = item['duration']
            traceid = item['output']["complete"][0]["data"]["trace_id"]
            action = (item["atomic_actions"][-1]['name'], item["atomic_actions"][-1]['started_at'], item["atomic_actions"][-1]['finished_at'], 1 if 'failed' in list(item["atomic_actions"][-1].keys()) else 0 )
            result.append([traceid, timesstamp, duration, error, *action])

    ground = pd.DataFrame(result, columns=['traceid', 'timestamp', 'duration', 'error', 'ac_name', 'ac_start', 'ac_end', 'ac_status'])
    return ground


def readtrace(data, base_trace=None, name='start', isfirst=False):
    if isfirst:        
        pass
    else:
        span_id = data['trace_id']
        parentspan_id = data['parent_id']
        info  = data['info']
        cmbd_id = info['host']
        spantype = info['name']
        try:
            start_event =  list(filter(lambda x: any('-start' in x for _ in list(info.keys())) , list(info.keys())))[0]
            span_start_time = info[start_event]['timestamp']
            end_event =  list(filter(lambda x: any('-stop' in x for _ in list(info.keys())) , list(info.keys())))[0]
            span_end_time = info[end_event]['timestamp']
        except:
            span_end_time = 'wrong'
        traceresult.append([cmbd_id, span_start_time, span_end_time, spantype, span_id, parentspan_id, base_trace, name])
        name = cmbd_id

    if data['children']:
        for span in data['children']:
            readtrace(span, base_trace, name)
    

def deal_log(data_path):
    config_file = "./util/msds.ini"
    template_file = "./util/log_template.csv"
    state_file = "./util/log_state.pkl"

    data = pd.read_csv(data_path, keep_default_na=False)
    data = data[['Hostname','@timestamp','log_level','Payload']]
    data['@timestamp'] = data['@timestamp'].map(lambda x: x[:-10])
    data['@timestamp'] = data['@timestamp'].map(lambda x: time.mktime(time.strptime(x, '%Y-%m-%dT%H:%M:%S.%f')) - 60*60)
    data = data.loc[(data['@timestamp'] >= start_dealtime) & (data['@timestamp'] <= end_dealtime + 1), :]

    template_miner = createparser(config_file, state_file)
    templateidlist = logparse(template_miner, data['Payload'])
    data.insert(loc=0, column='templateid', value=templateidlist)
    save_template(template_miner, template_file)
    save_state(template_miner, state_file)
    return data

def deal_kpi(data_path):
    result = []
    for item in os.listdir(data_path):
        nodename = item.split('_')[0]
        data = pd.read_csv(os.path.join(data_path, item), sep=',')
        data = data.groupby('now').mean()
        data = data[['cpu.user', 'mem.used', 'load.min1', 'load.min5', 'load.min15']]
        for name in data.columns.values:
            data[name] = (data[name] - data[name].min()) / (data[name].max() - data[name].min())
        data = data.fillna(0)
        data.columns = data.columns.map(lambda x: f'{nodename}_'+x)
        result.append(data)
    result = pd.concat(result, axis=1, ignore_index=False)
    result = result.dropna(axis=0, how='any')
    result.sort_index(axis=0)
    return result

print('deal metric')
metric = deal_kpi(os.path.join(raw_path,'metrics'))
print(metric.head())
metric.index = metric.index.map(lambda x: x[:-5])
metric.index = metric.index.map(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')) - 60*60)
metric = metric.loc[(metric.index >= start_dealtime) & (metric.index <= end_dealtime), :]
metric.to_csv(os.path.join(save_path, 'metric.csv'), index=True)

print('deal log')
log = deal_log(os.path.join(raw_path, 'logs/logs_aggregated_concurrent.csv'))
log.to_csv(os.path.join(save_path, 'log.csv'), sep=',', index=False, header=True)

print('deal trace')
traceresult = []
for path, dir_lst, file_lst in os.walk(os.path.join(raw_path, 'traces')):
    for file_name in tqdm(file_lst):
        data = json.load(open(os.path.join(path, file_name)))
        name = file_name.split('.')[0]
        readtrace(data, base_trace=name, name='start', isfirst=True)
trace = pd.DataFrame(traceresult, columns=['cmbd_id', 'start_time', 'end_time', 'stats', 'span_id', 'parentspan_id', 'base_trace', 'fatherpod'])
trace['start_time'], trace['start_sec'] = trace['start_time'].str.split('.').str
trace['start_time'] = trace['start_time'].map(lambda x: time.mktime(time.strptime(x, '%Y-%m-%dT%H:%M:%S')))
trace['end_time'].replace('wrong', np.nan, inplace=True)
trace['end_time'].fillna(method='ffill', inplace=True)
trace['end_time'], trace['end_sec'] = trace['end_time'].str.split('.').str
trace['end_time'] = trace['end_time'].map(lambda x: time.mktime(time.strptime(x, '%Y-%m-%dT%H:%M:%S')))

pod_relation = trace.apply(lambda x: '_'.join([x['fatherpod'],x['cmbd_id']]), axis=1)  
pod_relation = list(set(pod_relation))

relation_matrix = np.zeros((len(MSDS_pod), len(MSDS_pod)))
for item in pod_relation:
    [start, end] = item.split('_')
    if start not in MSDS_pod or end not in MSDS_pod:
        continue
    relation_matrix[MSDS_pod.index(start), MSDS_pod.index(end)] = 1
    relation_matrix[MSDS_pod.index(end), MSDS_pod.index(start)] = 1
pickle.dump(relation_matrix, open(os.path.join(save_path, 'trace_path.pkl'), 'wb'))

ground = deal_ground(os.path.join(raw_path, 'reports'))
ground.to_csv(os.path.join(save_path, 'groundtruth.csv'), index=None)
label = np.zeros((int((end_dealtime-start_dealtime)+ 1), len(MSDS_pod)))
#remove unlabel trace
trace_lack = list(set(trace['base_trace'].unique()).difference(set(ground['traceid'].values)))
index = []
for traceid in tqdm(trace_lack):
    index.extend(trace[(trace.base_trace == traceid)].index.to_list())
trace = trace.drop(index=list(set(index)))

trace[['end_time', 'start_time', 'end_sec', 'start_sec']] = trace[['end_time', 'start_time', 'end_sec', 'start_sec']].apply(pd.to_numeric)
trace['duration'] = trace['end_time'] - trace['start_time'] + (trace['end_sec'] - trace['start_sec']) * 1e-6
trace_mean = trace.groupby('stats')['duration'].mean()
trace['duration'] = trace.apply(lambda x: x['duration'] / trace_mean[x['stats']], axis=1)
trace.to_csv(os.path.join(save_path, 'trace.csv'), index=False)

ground = ground.loc[ground['error'] == 1]
ground[['ac_start', 'ac_end']] = ground[['ac_start', 'ac_end']].apply(pd.to_numeric)
for _, item in tqdm(ground.iterrows()):
    trace_data = trace.loc[trace['base_trace'] == item['traceid']]
    cmbd_list = trace_data['cmbd_id'].unique().tolist()
    cmbd_index = list(map(lambda x: MSDS_pod.index(x), cmbd_list))
    time_list = [item for item in range(int(item['ac_start']-start_dealtime), int(item['ac_end']-start_dealtime + 1))]
    for x in cmbd_index:
        label[time_list, x] = 1
pickle.dump(label, open(os.path.join(save_path, 'label.pkl'), 'wb'))
    

    
