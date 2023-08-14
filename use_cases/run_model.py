import sys
sys.path.append('../')

import mowl
mowl.init_jvm("10g")

import click as ck
import os
from src.owl2vec import OWL2Vec
from src.cate import CatE
from src.el_sem import ELModel

from src.utils import seed_everything
import logging



@ck.command()
@ck.option('--use-case', '-case', required=True, type=ck.Choice(["ore1", "ore2", "ore3", "owl2bench1", "owl2bench2", "caligraph4", "caligraph5"]))
@ck.option('--model-type', '-model', required=True, type=ck.Choice(['owl2vec', "cat", "elem", "elbox", "box2el"]))
@ck.option('--kge-model', '-kge', type=ck.Choice(['transe', 'ordere', "transd", "distmult", "conve"]), default='transe')
@ck.option('--root', '-r', required=True, type=ck.Path(exists=True))
@ck.option('--num-models', '-nm', default=1)
@ck.option('--emb-dim', '-dim', required=True, type=int, default=256)
@ck.option('--margin', '-m', required=True, type=float, default=0.1)
@ck.option('--weight-decay', '-wd', required=True, type=float, default = 0.0)
@ck.option('--batch-size', '-bs', required=True, type=int, default=4096*8)
@ck.option('--lr', '-lr', required=True, type=float, default=0.0001)
@ck.option('--num-negs', '-negs', required=True, type=int, default=4)
@ck.option('--test-batch-size', '-tbs', required=True, type=int, default=32)
@ck.option('--epochs', '-e', required=True, type=int, default=300)
@ck.option('--device', '-d', required=True, type=ck.Choice(['cpu', 'cuda']))
@ck.option('--seed', '-s', required=True, type=int, default=42)
@ck.option('--only_sub', '-sub', is_flag=True)
@ck.option('--only_mem', '-mem', is_flag=True)
@ck.option("--only_train", '-otr', is_flag=True)
@ck.option("--only_test", '-ot', is_flag=True)
@ck.option("--aggregator", '-agg', type=ck.Choice(['mean', 'min', 'max', 'median']), default='mean')
@ck.option('--result-filename', '-rf', required=True)
def main(use_case, model_type, kge_model, root, num_models,
         emb_dim, margin, weight_decay, batch_size, lr, num_negs,
         test_batch_size, epochs, device, seed, only_sub, only_mem, only_train, only_test, aggregator, result_filename):

    if not result_filename.endswith('.csv'):
        raise ValueError("For convenience, please specify a csv file as result_filename")

    if root.endswith('/'):
        root = root[:-1]

    #get parent of root
    root_parent = os.path.dirname(root)
        
    models_dir = os.path.join(root_parent, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
        
    print("Configuration:")
    print("\tuse_case: ", use_case)
    print("\tmodel_type: ", model_type)
    print("\tkge_model: ", kge_model)
    print("\troot: ", root)
    print("\tnum_models: ", num_models)
    print("\temb_dim: ", emb_dim)
    print("\tmargin: ", margin)
    print("\tweight_decay: ", weight_decay)
    print("\tbatch_size: ", batch_size)
    print("\tlr: ", lr)
    print("\tnum_negs: ", num_negs)
    print("\ttest_batch_size: ", test_batch_size)
    print("\tepochs: ", epochs)
    print("\tdevice: ", device)
    print("\tseed: ", seed)
    print("\tonly_train: ", only_train)
    print("\tonly_test: ", only_test)
    print("\taggregator: ", aggregator)
    print("\tonly_sub: ", only_sub)
    print("\tonly_mem: ", only_mem)
    print("\tresult_filename: ", result_filename)
    seed_everything(seed)

    if model_type in ['owl2vec', 'cat']:
        if model_type == "owl2vec":
            graph_model = OWL2Vec
        elif model_type == 'cat':
            graph_model = CatE
        
        model = graph_model(use_case,
                        kge_model,
                        root,
                        emb_dim,
                        margin,
                        weight_decay,
                        batch_size,
                        lr,
                        num_negs,
                        test_batch_size,
                        epochs,
                        device,
                        seed,
                        5, #tolerance,
                        )

    elif model_type in ["elem", "elbox", "box2el"]:
        model_filepath = os.path.join(models_dir, f"{use_case}_{model_type}_sub_{only_sub}_mem_{only_mem}.pt")
        model = ELModel(use_case,
                        model_type,
                        root,
                        num_models,
                        emb_dim,
                        margin,
                        batch_size,
                        lr,
                        test_batch_size,
                        epochs,
                        model_filepath,
                        device,
                        only_sub,
                        only_mem,
                        aggregator=aggregator)
        
        
    if not only_test:
        model.train()

    if not only_train:
        params = (emb_dim, margin, weight_decay, batch_size, lr, num_negs)
        
        subsumption_metrics, membership_metrics = model.test()
        print("Subsumption metrics: ")
        save_results(params, subsumption_metrics, result_filename)
        print("Membership metrics: ")
        save_results(params, membership_metrics, result_filename)
                                            
def save_results(params, metrics, result_dir):
    raw_metrics, filtered_metrics = metrics
    emb_dim, margin, weight_decay, batch_size, lr, num_negs = params
    mr, mrr, median_rank, h1, h3, h10, h100, auc, perc90, below1000 = raw_metrics
    mr_f, mrr_f, fmedian_rank, h1_f, h3_f, h10_f, h100_f, auc_f, fperc90, fbelow1000 = filtered_metrics
    with open(result_dir, 'a') as f:
        line1 = [emb_dim, margin, weight_decay, batch_size]
        line2 = [mrr, mr, median_rank, h1, h3, h10, h100, auc, perc90, below1000]
        line3 = [mrr_f, mr_f, fmedian_rank, h1_f, h3_f, h10_f, h100_f, auc_f, fperc90, fbelow1000]
        line = "|".join([str(x) for x in line1])
        line += "|" + "|".join([f"{x:.4f}" for x in line2])
        line += "|" + "|".join([f"{x:.4f}" for x in line3])
        line += "\n"
        f.write(line)
    print("Results saved to ", result_dir)
        
if __name__ == "__main__":
    main()




 
