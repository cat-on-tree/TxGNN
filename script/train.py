from txgnn import TxData, TxGNN
import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--split', type=str,
                    choices=['random', 'complex_disease', 'complex_disease_cv', 'disease_eval', 'cell_proliferation',
                             'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland', 'autoimmune',
                             'metabolic_disorder', 'diabetes', 'neurodigenerative', 'full_graph', 'downstream_pred',
                             'few_edeges_to_kg'])
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model', type=str, default='TxGNN', choices=['TxGNN', 'GNN', 'TxGAT', 'HGT'])

# Optional official pretrained TxGNN checkpoint directory.
# This directory should contain config.pkl and model.pt.
parser.add_argument('--pretrained_path', type=str, default=None)

# For local CPU/debug runs.
parser.add_argument('--data_folder', type=str, default='./data')
parser.add_argument('--save_root', type=str, default='./saved_models')
parser.add_argument('--finetune_epochs', type=int, default=500)
parser.add_argument('--finetune_lr', type=float, default=5e-4)
parser.add_argument('--train_print_per_n', type=int, default=5)
parser.add_argument('--valid_per_n', type=int, default=20)

args = parser.parse_args()

if args.device.lower() == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:' + str(args.device)

seed = args.seed
TxData = TxData(data_folder_path=args.data_folder)
TxData.prepare_split(split=args.split, seed=seed, no_kg=False)

name = '_'.join([args.model, str(args.seed), args.split])
if args.pretrained_path is not None:
    name = name + '_pretrained'

TxGNN = TxGNN(data=TxData,
              weight_bias_track=False,
              proj_name='TxGNN_Baselines',
              exp_name=name,
              device=device
              )

# Load official pretrained TxGNN and skip model_initialize/pretrain.
if args.pretrained_path is not None:
    if args.model != 'TxGNN':
        raise ValueError('--pretrained_path is only supported for --model TxGNN.')

    print('Loading pretrained TxGNN from:', args.pretrained_path)
    TxGNN.load_pretrained(args.pretrained_path)

else:
    if args.model in ["GNN", "HGT"]:
        proto = False
    else:
        proto = True

    if args.model == "TxGAT":
        attention = True
    else:
        attention = False

    if args.model == "HGT":
        model_type = "hgt"
    else:
        model_type = "txgnn"

    TxGNN.model_initialize(n_hid=100,
                           n_inp=100,
                           n_out=100,
                           proto=proto,
                           proto_num=3,
                           attention=attention,
                           sim_measure='all_nodes_profile',
                           agg_measure='rarity',
                           model_type=model_type,
                           hgt_num_heads=4,
                           hgt_dropout=0.2,
                           hgt_use_norm=True)

TxGNN.finetune(n_epoch=args.finetune_epochs,
               learning_rate=args.finetune_lr,
               train_print_per_n=args.train_print_per_n,
               valid_per_n=args.valid_per_n)

TxGNN.save_model(args.save_root + '/' + name)