import argparse
import subprocess
import time
import csv

gat_result = dict()
gcn_result = dict()
appnp_result = dict()

final_result = []

class Exp():
    def __init__(self, exp_list):
        self.config_list = exp_list
        self.result = dict() 
        self.is_hetero = False

    def gen_launch_configs(self):
        configs = []
        def enumerate(depth, config):
            if depth == len(self.config_list):
                configs.append(config)
            else:
                k, conf_list = self.config_list[depth]
                for con in conf_list:
                    enumerate(depth+1, config + self.build_arg(k, con))
        enumerate(0, '')
        return configs
    
    def build_arg(self, k, v):
        '''
          To simplify launch configrations, all systems must implement a train.py to run the model,
          which use argparse library and can be run using python trian.py --gpu 0 --num_layers 2. (DGL is a good example).
        '''
        return ' --{key} {val} '.format(key=k, val=v)
   
    def post_processing(self, model, system, dataset, output):
        '''
          We need to print some keywords in every model. Search those keywords in output to 
          get the required results such as memory usage and mini-batch time
        '''
        output_str = str(output, 'utf-8')
        split_list = output_str.split('^^^')
        print('Used memory(GB)', split_list[1], 'Epoch time(s)', split_list[2])

        if len(split_list) >=3:
            run_time = float(split_list[2])
            memory = float(split_list[1])
        else :
            run_time = -1.0
            memory = -1.0

        final_result.append([model, system, dataset, memory, run_time])
        '''
        if model == 'gcn':
            print('gcn_', memory, run_time)
            gcn_result[(system, dataset)] = (memory, run_time)
            print('gcn_result = ', gcn_result)
        elif model == 'gat':
            print('gat_', memory, run_time)
            gat_result[(system, dataset)] = (memory, run_time)
            print('gat_result = ', gat_result)
        elif model == 'appnp':
            print('appnp_', memory, run_time)
            appnp_result[(system, dataset)] = (memory, run_time)
            print('appnp_result = ', appnp_result)
        else :
            print('no such
             model', model)
        '''

        #final_result[(model, system)] = (memory, run_time)
        #print(final_result)

    def get_dataset_name(self, config):
        config_list = config.split(' ')
        #print("config_list = ", config_list) 
        for idx in range(len(config_list)):
            if config_list[idx] == '--dataset':
                dataset_name = str(config_list[idx+1])
                #print('dataset_name = ', dataset_name)
                return dataset_name
    
    def _execute(self, sys, config):
        try:
            script_path = './{model_name}/{system}/train.py'.format(model_name=self.model_name, system=sys)
            cmd = 'CUDA_VISIBLE_DEVICES=0 python {script_path} {config}'.format(script_path=script_path, config=config)
            dataset_name = self.get_dataset_name(config)
            print('to execute cmd:', cmd)
            output = subprocess.check_output(cmd, shell=True)
            self.post_processing(str(self.model_name), str(sys), dataset_name, output)
        except Exception as exe:
            pass

    def run_on_systems(self, sys_list):
        for config in self.gen_launch_configs():
            for sys in sys_list: 
                self._execute(sys, config)
                if self.is_hetero and sys != 'seastar':
                    self._execute(str(sys)+'-hetero', config)

#num_layers : num_hidden layers. actually '(num_layers + 1)' layers
class GATExp(Exp):
    '''
      Write as a class to restrict the set of tunable parameters to be those declared in constructors
      to avoid incidently pass un-recognizable parameter to models
      Add more tunable parameters for gat in the constructor
    '''
    def __init__(self, dataset, num_heads, num_hidden, num_layers, gpu, num_epochs):
        self.config_list = list({
            'gpu' : gpu,
            'num_epochs': num_epochs,
            'dataset': list(dataset),
            'num_heads': list(num_heads),
            'num_hidden': list(num_hidden),
            'num_layers': list(num_layers),
        }.items())
        Exp.__init__(self, self.config_list)
        self.model_name = 'gat'

class GCNExp(Exp):
    '''
      Write as a class to restrict the set of tunable parameters to be those declared in constructors
      to avoid incidently pass un-recognizable parameter to models
      Add more tunable parameters for GCN in the constructor
    '''
    def __init__(self, dataset, num_hidden, num_layers, gpu, num_epochs):
        self.config_list = list({
            'gpu' : gpu,
            'num_epochs': num_epochs,
            'dataset': list(dataset),
            'num_hidden': list(num_hidden), 
            'num_layers': list(num_layers),
        }.items())
        Exp.__init__(self, self.config_list)
        self.model_name = 'gcn'

class APPNPExp(Exp):
    '''
      Write as a class to restrict the set of tunable parameters to be those declared in constructors
      to avoid incidently pass un-recognizable parameter to models
      Add more tunable parameters for APPNP in the constructor
    '''
    def __init__(self, dataset, hidden_sizes, k, gpu, num_epochs):
        self.config_list = list({
            'gpu' : gpu,
            'num_epochs': num_epochs,
            'dataset': list(dataset),
            'hidden_sizes': list(hidden_sizes), 
            'k': list(k),
        }.items())
        Exp.__init__(self, self.config_list)
        self.model_name = 'appnp'
    
class RGCNExp(Exp):
    def __init__(self, dataset, hidden_size, num_bases, gpu, num_epochs):
        self.config_list = list({
            'dataset' : list(dataset),
            'hidden_size' : list(hidden_size),
            'num_bases': list(num_bases),
            'gpu': gpu,
            'num_epochs' : num_epochs,
        }.items())
        Exp.__init__(self, self.config_list)
        self.is_hetero = True
        self.model_name = 'rgcn'

def create_exp_list_sample0(args):
    exp_list = []

    #homo_model_dataset = ['reddit']
    homo_model_dataset=['cora', 'citeseer', 'pubmed', 'CoraFull', 'Coauthor_cs', 'Coauthor_physics', 'AmazonCoBuy_photo', 'AmazonCoBuy_computers', 'reddit']
    hetero_model_dataset=['aifb', 'mutag', 'bgs']
    #homo_model_dataset=['CoraFull', 'Coauthor_cs', 'Coauthor_physics', 'AmazonCoBuy_photo', 'AmazonCoBuy_computers', 'reddit']
                    
    for model in args.models:
        if model == 'gat':
            exp_list.append(
                GATExp(
                    dataset=homo_model_dataset,
                    num_heads=[8],
                    num_hidden=[32],
                    num_layers=[1],
                    gpu=[args.gpu],
                    num_epochs=[args.num_epochs])
            )
        elif 'gcn' == model:
            exp_list.append(
                GCNExp(
                    dataset=homo_model_dataset,
                    num_hidden=[32],
                    num_layers=[1],
                    gpu=[args.gpu],
                    num_epochs=[args.num_epochs])
            )
        elif 'appnp' == model:
            exp_list.append(
                APPNPExp(
                    dataset=homo_model_dataset,
                    hidden_sizes=[64],
                    k=[10],
                    gpu=[args.gpu],
                    num_epochs=[args.num_epochs])
            )
        elif 'rgcn' == model:
            exp_list.append(
                RGCNExp(
                    dataset=hetero_model_dataset,
                    hidden_size=[16],
                    num_bases=[40],
                    gpu=[args.gpu],
                    num_epochs=[args.num_epochs]
                )
            )
    assert len(exp_list) >= 1 and len(args.systems) >= 1, 'Cannot find experiment to run'
    return exp_list

def result_write_to_file(result_dict, path):

    for key, value in result_dict.items():
        print(key, value)
        
    s = str(result_dict)
    f = open(path, 'w')
    f.writelines(s)
    f.close()

def main(args):

    print(args.models, args.systems, args.gpu, args.num_epochs)
    exp_list = create_exp_list_sample0(args)

    for exp in exp_list:
        exp.run_on_systems(args.systems)

    print(final_result)
    with open('./result/final_result.csv', 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)
        print("write to ./result/final_result.csv")
        writer.writerows(final_result)
    #write csv
    '''
    result_write_to_file(gat_result, './result/gat_result.txt')
    result_write_to_file(gcn_result, './result/gcn_result.txt')
    result_write_to_file(appnp_result, './result/appnp_result.txt')

    
    for key, value in final_result.items():
        print(key, value)
        
    s = str(final_result)
    f = open('result.txt', 'w')
    f.writelines(s)
    f.close()
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument("--models", nargs='+', default='gat',
                        help="which models to run. Example usage: --models gat gcn")
    parser.add_argument("--systems", nargs='+', default='dgl',
                        help="which models to run. Example usage: --systems dgl seastar pyg")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="number of training epochs")
    args = parser.parse_args()
    args.models = [args.models] if not isinstance(args.models, list) else args.models
    args.systems = [args.systems] if not isinstance(args.systems, list) else args.systems
    print(args)
    main(args)
