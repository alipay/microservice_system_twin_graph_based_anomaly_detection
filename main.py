import util.util as util
import util.train as train
import util.data_MSDS as data_loads
from util.parser_MSDS import *

from torch.utils.data import DataLoader
import warnings
import logging
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append('/code')
warnings.filterwarnings("ignore")

util.seed_everything(args['random_seed'])

if __name__ == '__main__':
    if args['evaluate']:
        dict_json = util.read_params(args)
        for key in dict_json.keys():
            args[key] =  args[key] if key in ['model_path','evaluate', 'result_dir', 'data_path', 'dataset_path'] else dict_json[key]
        args['result_dir'] = args['model_path']
    else:
        args['hash_id'], args['result_dir'] = util.dump_params(args)
        util.json_pretty_dump(args, os.path.join(args['result_dir'], "params.json"))
        args['model_path'] = args['result_dir']

    logging.info("---- Model: ----" + args['main_model'] +"-" + args['hash_id'] + "----" + f"train : {not args['evaluate']}"\
        + "----" + f"evaluate : {args['evaluate']}")

    # dealing & loading data
    processed = data_loads.Process(**args)
    train_dl = DataLoader(processed.dataset[:int(len(processed.dataset)*0.7)],
                          batch_size=args['batch_size'],
                          shuffle=True, pin_memory=False, drop_last=True)
    test_dl = DataLoader(processed.dataset[int(len(processed.dataset)*0.7):],
                        batch_size=args['batch_size'],
                        shuffle=False, pin_memory=False, drop_last=True)

    # declear model and train
    import src.model as model
    models = model.MyModel(processed.graph, **args)
    sys = train.MY(models, **args)  

    #Training
    if not args['evaluate']:
        sys.fit(train_loader=train_dl, test_loader=test_dl)


    # Evaluating
    logging.info('calculate scores...')
    with open('./result.log', 'a+') as file:
        file.writelines(f"\n {args['main_model']}-{args['hash_id']} --weight_decay:{args['weight_decay']}   --learning_change:{args['learning_change']} \n")
        for statue in ['loss', 'f1']:
            logging.info(f'calculate label with {statue}...')
            sys.load_model(args['model_path'], name=statue)
            info = sys.evaluate(test_dl, isFinall=True)
            file.writelines(statue + '   ' + info + '\n')
    logging.info("^^^^^^ Current Model: ----" + args['main_model'] + "-" * 4 + args['hash_id'] + " ^^^^^")

