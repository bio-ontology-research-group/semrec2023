from mowl.base_models.elmodel import EmbeddingELModel
from mowl.projection.factory import projector_factory
from tqdm import trange, tqdm
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import os
from itertools import cycle
import sys

from mowl.base_models import EmbeddingELModel
from mowl.nn import ELEmModule, ELBoxModule, BoxSquaredELModule
from mowl.utils.data import FastTensorDataLoader
from mowl.datasets import PathDataset, Dataset
from mowl.owlapi import OWLAPIAdapter


from org.semanticweb.owlapi.model.parameters import Imports
from java.util import HashSet


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ELModule(nn.Module):
    def __init__(self, module_name, dim, nb_classes, nb_individuals, nb_roles):
        super().__init__()

        self.module_name = module_name
        self.dim = dim
        self.nb_classes = nb_classes
        self.nb_individuals = nb_individuals
        self.nb_roles = nb_roles

        self.set_module(self.module_name)

        self.ind_embeddings = nn.Embedding(self.nb_individuals, self.dim)


    def set_module(self, module_name):
        if module_name == "elem":
            self.el_module = ELEmModule(self.nb_classes, self.nb_roles, embed_dim = self.dim)
        elif module_name == "elbox":
            self.el_module = ELBoxModule(self.nb_classes, self.nb_roles, embed_dim = self.dim)
        elif module_name == "box2el":
            self.el_module = BoxSquaredELModule(self.nb_classes, self.nb_roles, embed_dim = self.dim)
        else:
            raise ValueError("Unknown module: {}".format(module))

    def tbox_forward(self, *args, **kwargs):
        return self.el_module(*args, **kwargs)

    def abox_forward(self, ind_idxs):
        class_embed = self.el_module.class_center if self.module_name == "box2el" else self.el_module.class_embed
        all_class_embed = class_embed.weight
        ind_embed = self.ind_embeddings(ind_idxs)

        membership = th.mm(ind_embed, all_class_embed.t())

        if self.module_name == "elem":
            rad_embed = self.el_module.class_rad.weight
            rad_embed = th.abs(rad_embed).view(1, -1)
            membership = membership + rad_embed
        elif self.module_name in ["elbox", "box2el"]:
            offset_embed = self.el_module.class_offset.weight
            offset_embed = th.abs(offset_embed).mean(dim=1).view(1, -1)
            membership = membership + offset_embed
        else:
            raise ValueError("Unknown module: {}".format(self.module_name))

        return membership

class ELModel(EmbeddingELModel) :
    

    def __init__(self, use_case, model_name, root, num_models, embed_dim, margin, batch_size, lr,
                 test_batch_size, epochs, model_filepath, device, only_sub, only_mem, aggregator="mean"):

        self.module_name = model_name
        self.root= root
        self.margin = margin
        self.num_models = num_models
        self.learning_rate = lr
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.only_sub = only_sub
        self.only_mem = only_mem
        self.aggregator = aggregator
        
        if not model_filepath.endswith(".pt"):
            raise ValueError("Model filepath must end with .pt")
        
        if "ore" in use_case:
            number = use_case[-1]
            train_path = os.path.join(self.root, f"ORE{number}_cleaned.owl") # given and modified a priori
            valid_path = os.path.join(self.root, f"_valid_ORE{number}_wrapped.owl") # given
            test_path = os.path.join(self.root, f"_test_ORE{number}_wrapped.owl") # given
        elif "owl2bench" in use_case:
            number = use_case[-1]
            train_path = os.path.join(self.root, f"OWL2DL-{number}_cleaned.owl")
            valid_path = os.path.join(self.root, f"_valid_OWL2Bench{number}_wrapped.owl")
            test_path = os.path.join(self.root, f"_test_OWL2Bench{number}_wrapped.owl")
        elif "caligraph" in use_case:
            number = use_case[-1]
            train_path = os.path.join(self.root, f"clg_10e{number}_cleaned.owl")
            valid_path = os.path.join(self.root, f"clg_10e{number}-val_corrected.owl")
            test_path = os.path.join(self.root, f"clg_10e{number}-test_corrected.owl")
            
        dataset = PathDataset(train_path, validation_path=valid_path, testing_path=test_path)
                        
        train_ont = self.preprocess_ontology(dataset.ontology)
        valid_ont = self.preprocess_ontology(dataset.validation)
        test_ont = self.preprocess_ontology(dataset.testing)

        dataset = Dataset(train_ont, validation=valid_ont, testing=test_ont)
                                
        super().__init__(dataset, embed_dim, batch_size, model_filepath=model_filepath,  device=device)
        self.init_modules()

    def preprocess_ontology(self, ontology):
        """Preprocesses the ontology to remove axioms that are not supported by the normalization \
            process.

        :param ontology: Input ontology
        :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`

        :rtype: :class:`org.semanticweb.owlapi.model.OWLOntology`
        """

        tbox_axioms = ontology.getTBoxAxioms(Imports.fromBoolean(True))
        abox_axioms = ontology.getABoxAxioms(Imports.fromBoolean(True))
        new_tbox_axioms = HashSet()
        for axiom in tbox_axioms:
            axiom_as_str = axiom.toString()

            if "ObjectHasValue" in axiom_as_str:
                continue
            elif "DataSomeValuesFrom" in axiom_as_str:
                continue
            elif "DataAllValuesFrom" in axiom_as_str:
                continue
            elif "DataHasValue" in axiom_as_str:
                continue
            elif "DataPropertyRange" in axiom_as_str:
                continue
            elif "DataPropertyDomain" in axiom_as_str:
                continue
            elif "FunctionalDataProperty" in axiom_as_str:
                continue
            elif "DisjointUnion" in axiom_as_str:
                continue
            elif "HasKey" in axiom_as_str:
                continue
            
            new_tbox_axioms.add(axiom)

        owl_manager = OWLAPIAdapter().owl_manager
        new_ontology = owl_manager.createOntology(new_tbox_axioms)
        new_ontology.addAxioms(abox_axioms)
        return new_ontology

        
    def init_modules(self):
        logger.info("Initializing modules...")
        nb_classes = len(self.dataset.classes)
        nb_individuals = len(self.dataset.individuals)
        nb_roles = len(self.dataset.object_properties)
        modules = []

        for i in range(self.num_models):
            module = ELModule(self.module_name, self.embed_dim, nb_classes, nb_individuals, nb_roles)
            modules.append(module)
        self.modules = nn.ModuleList(modules)
        logger.info(f"Created {len(self.modules)} modules")
        
    def is_named_class(self, class_expression):
        return class_expression.getClassExpressionType() == ClassExpressionType.OWL_CLASS

    def get_abox_data(self, ont):
        if ont == "train":
            ontology = self.dataset.ontology
        elif ont == "valid":
            ontology = self.dataset.validation
        elif ont == "test":
            ontology = self.dataset.testing
        else:
            raise ValueError("Unknown ontology: {}".format(ont))

        abox = []
        for cls in self.dataset.classes:
            abox.extend(list(ontology.getClassAssertionAxioms(cls)))
        
        # abox = [axiom for axiom in train_abox if self.is_named_class(axiom.getClassExpression())]

        nb_inds = len(self.dataset.individuals)
        assert nb_inds > 0
        nb_classes = len(self.dataset.classes)

        owlind_to_id = self.dataset.individuals.to_index_dict()
        owlclass_to_id = self.dataset.classes.to_index_dict()

        
        labels = np.zeros((nb_inds, nb_classes), dtype=np.int32)

        for axiom in abox:
            cls = axiom.getClassExpression()
            ind = axiom.getIndividual()

            cls_id = owlclass_to_id[cls]
            ind_id = owlind_to_id[ind]
            labels[ind_id, cls_id] = 1

        idxs = np.arange(nb_inds)
        return th.tensor(idxs), th.FloatTensor(labels)
        

    
        
    def train(self):

        abox_ds_train = self.get_abox_data("train")
        abox_ds_valid = self.get_abox_data("valid")
        
        abox_dl_train = FastTensorDataLoader(*abox_ds_train, batch_size=self.batch_size, shuffle=True)
        abox_dl_valid = FastTensorDataLoader(*abox_ds_valid, batch_size=self.batch_size, shuffle=False)
        
        el_dls = {gci_name: DataLoader(ds, batch_size=self.batch_size, shuffle=True) for gci_name, ds in self.training_datasets.items() if len(ds) > 0}
        el_dls_sizes = {gci_name: len(ds) for gci_name, ds in self.training_datasets.items() if len(ds) > 0}

        if len(self.dataset.individuals) > el_dls_sizes["gci0"]:
            main_dl = abox_dl_train
            main_dl_name = "abox"
        else:
            main_dl = el_dls["gci0"]
            main_dl_name = "gci0"

        print("Main DataLoader: {}".format(main_dl_name))
            
        total_el_dls_size = sum(el_dls_sizes.values())
        el_dls_weights = {gci_name: ds_size / total_el_dls_size for gci_name, ds_size in el_dls_sizes.items()}

        if main_dl_name == "gci0":
            el_dls = {gci_name: cycle(dl) for gci_name, dl in el_dls.items() if gci_name != "gci0"}
            abox_dl_train = cycle(abox_dl_train)
        else:
            el_dls = {gci_name: cycle(dl) for gci_name, dl in el_dls.items()}
        logger.info(f"Dataloaders: {el_dls_sizes}")

        tolerance = 10
        current_tolerance = tolerance

        nb_classes = len(self.dataset.classes)

        for i, module in enumerate(self.modules):
            logger.info(f"Training module {i+1}/{len(self.modules)}")
            sub_module_filepath = self.model_filepath.replace(".pt", f"_{i+1}_of_{len(self.modules)}.pt")
            
            optimizer = th.optim.Adam(module.parameters(), lr=self.learning_rate)
            min_lr = self.learning_rate / 10
            max_lr = self.learning_rate
            scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=20, cycle_momentum=False)
            best_loss = float('inf')
            best_mr = float('inf')
                                                                        
            for epoch in trange(self.epochs):
                module = module.to(self.device)
                module.train()
                

                train_el_loss = 0
                train_abox_loss = 0

                for batch_data in main_dl:
                    if main_dl_name == "abox":
                        ind_idxs, labels = batch_data
                        gci0_batch = next(el_dls["gci0"]).to(self.device)
                    elif main_dl_name == "gci0":
                        gci0_batch = batch_data.to(self.device)
                        ind_idxs, labels = next(abox_dl_train)

                    pos_gci0 = module.tbox_forward(gci0_batch, "gci0").mean() * el_dls_weights["gci0"]
                    #el_loss = pos_gci0
                    neg_idxs = np.random.choice(nb_classes, size=len(gci0_batch), replace=True)
                    neg_batch = th.tensor(neg_idxs, dtype=th.long, device=self.device)
                    neg_data = th.cat((gci0_batch[:, :1], neg_batch.unsqueeze(1)), dim=1)
                    neg_gci0 = module.tbox_forward(neg_data, "gci0").mean() * el_dls_weights["gci0"]
                    el_loss = -F.logsigmoid(-pos_gci0 + neg_gci0 - self.margin).mean()
                    
                    for gci_name, gci_dl in el_dls.items():
                        if gci_name == "gci0":
                            continue

                        gci_batch = next(gci_dl).to(self.device)
                        pos_gci = module.tbox_forward(gci_batch, gci_name).mean() * el_dls_weights[gci_name]
                        neg_idxs = np.random.choice(nb_classes, size=len(gci_batch), replace=True)
                        neg_batch = th.tensor(neg_idxs, dtype=th.long, device=self.device)
                        neg_data = th.cat((gci_batch[:, :2], neg_batch.unsqueeze(1)), dim=1)
                        neg_gci = module.tbox_forward(neg_data, gci_name).mean() * el_dls_weights[gci_name]

                        el_loss += -F.logsigmoid(-pos_gci + neg_gci - self.margin).mean()
                        
                    abox_logits = module.abox_forward(ind_idxs.to(self.device))
                    abox_loss = F.binary_cross_entropy_with_logits(abox_logits, labels.to(self.device))

                    loss = el_loss + abox_loss
                                                                                                 
                    loss.backward()
                    optimizer.step()
                    train_el_loss += el_loss.item()
                    train_abox_loss += abox_loss.item()

                train_el_loss /= len(main_dl)
                train_abox_loss /= len(main_dl)

                valid_subsumption_mr = self.compute_ranking_metrics(mode="valid_subsumption", model=i)
                valid_membership_mr = self.compute_ranking_metrics(mode="valid_membership", model=i)
                if self.only_sub and self.only_mem:
                    total_mr = (valid_subsumption_mr + valid_membership_mr)
                    sub_weight = 0.5 #valid_subsumption_mr/total_mr
                    mem_weight = 0.5 #valid_membership_mr/total_mr
                    valid_mr = sub_weight*valid_subsumption_mr + mem_weight*valid_membership_mr
                elif self.only_sub:
                    valid_mr = valid_subsumption_mr
                elif self.only_mem:
                    valid_mr = valid_membership_mr
                    
                if valid_mr < best_mr:
                    best_mr = valid_mr
                    current_tolerance = tolerance+1
                    th.save(module.state_dict(), sub_module_filepath)
                    print("Model saved")
                print(f"Training: EL loss: {train_el_loss:.6f} | ABox loss: {train_abox_loss:.6f} | Valid MR: {valid_mr:.6f} | Valid subsumption MR: {valid_subsumption_mr:.6f} | Valid membership MR: {valid_membership_mr:.6f}")

                current_tolerance -= 1
                if current_tolerance == 0:
                    print("Early stopping")
                    break

                                

                
    def get_filtering_labels(self):
        logger.info("Getting predictions and labels")

        num_test_subsumption_heads = len(self.dataset.classes)
        num_test_subsumption_tails = len(self.dataset.classes)
        num_test_membership_heads = len(self.dataset.individuals)
        num_test_membership_tails = len(self.dataset.classes)
        
        # get predictions and labels for test subsumption
        filter_subsumption_labels = np.ones((num_test_subsumption_heads, num_test_subsumption_tails), dtype=np.int32)
        filter_membership_labels = np.ones((num_test_membership_heads, num_test_membership_tails), dtype=np.int32)
        
        classes_ids = th.arange(len(self.dataset.classes), device=self.device)
        individuals_ids = th.arange(len(self.dataset.individuals), device=self.device)


        ds = self.training_datasets["gci0"][:]
        sub = ds[:, 0]
        super_ = ds[:, 1]
        subsumption_dl = FastTensorDataLoader(sub, super_, batch_size=self.test_batch_size, shuffle=False)
        _, labels = self.get_abox_data("train")
        ds = th.nonzero(labels).squeeze()
        inds = ds[:, 0]
        cls = ds[:, 1]
        
        membership_dl = FastTensorDataLoader(inds, cls, batch_size=self.test_batch_size, shuffle=False)
            
        with th.no_grad():
            for head_idxs, tail_idxs in tqdm(subsumption_dl, desc="Getting labels"):
                head_idxs = head_idxs.cpu().numpy()
                tail_idxs = tail_idxs.cpu().numpy()
                filter_subsumption_labels[head_idxs, tail_idxs] = 10000

            for head_idxs, tail_idxs in tqdm(membership_dl, desc="Getting labels"):
                head_idxs = head_idxs.cpu().numpy()
                tail_idxs = tail_idxs.cpu().numpy()
                filter_membership_labels[head_idxs, tail_idxs] = 10000
        return filter_subsumption_labels, filter_membership_labels



    def compute_ranking_metrics(self, filtering_labels=None, mode="test_subsumption", model=None):
        if not mode in ["test_subsumption", "valid_subsumption", "test_membership", "valid_membership"]:
            raise ValueError(f"Invalid mode {mode}")

        if filtering_labels is None and "test" in mode:
            raise ValueError("filtering_labels cannot be None when mode is test")

        if filtering_labels is not None and "validate" in mode:
            raise ValueError("filtering_labels must be None when mode is validate")


        if "test" in mode:
            self.load_best_model()

        all_tail_ids = th.arange(len(self.dataset.classes)).to(self.device)
        if "subsumption" in mode:
            all_head_ids = th.arange(len(self.dataset.classes)).to(self.device)
            ds = self.testing_datasets["gci0"][:]
            sub = ds[:, 0]
            super_ = ds[:, 1]
            eval_dl = FastTensorDataLoader(sub, super_, batch_size=self.test_batch_size, shuffle=False)
        elif "membership" in mode:
            all_head_ids = th.arange(len(self.dataset.individuals)).to(self.device)
            if "test" in mode:
                _, labels = self.get_abox_data("test")
            elif "valid" in mode:
                _, labels = self.get_abox_data("valid")
            else:
                raise ValueError(f"Invalid mode {mode}")
        
            ds = th.nonzero(labels).squeeze()
            inds = ds[:, 0]
            cls = ds[:, 1]
            eval_dl = FastTensorDataLoader(inds, cls, batch_size=self.test_batch_size, shuffle=False)
            
            
        mean_rank, filtered_mean_rank = 0, 0
        ranks, filtered_ranks = dict(), dict()
        rank_vals = []
        filtered_rank_vals = []
        if "test" in mode:
            mrr, filtered_mrr = 0, 0
            hits_at_1, fhits_at_1 = 0, 0
            hits_at_3, fhits_at_3 = 0, 0
            hits_at_10, fhits_at_10 = 0, 0
            hits_at_100, fhits_at_100 = 0, 0

        with th.no_grad():
            for head_idxs, tail_idxs in eval_dl:

                predictions = self.predict(head_idxs, tail_idxs, mode=mode, model=model)
                
                assert predictions.shape[0] == head_idxs.shape[0], f"Predictions shape: {predictions.shape}, head_idxs shape: {head_idxs.shape}"
                
                for i, head in enumerate(head_idxs):
                    tail = tail_idxs[i]
                    preds = predictions[i]

                    orderings = th.argsort(preds, descending=True)
                    rank = th.where(orderings == tail)[0].item()
                    mean_rank += rank
                    rank_vals.append(rank)
                    if rank not in ranks:
                        ranks[rank] = 0
                    ranks[rank] += 1

                    if "test" in mode:
                        
                        filt_labels = filtering_labels[head, :]
                        filt_labels[tail] = 1
                        filtered_preds = preds.cpu().numpy() * filt_labels
                        filtered_preds = th.from_numpy(filtered_preds).to(self.device)
                        filtered_orderings = th.argsort(filtered_preds, descending=True) 
                        filtered_rank = th.where(filtered_orderings == tail)[0].item()
                        filtered_mean_rank += filtered_rank
                        filtered_rank_vals.append(filtered_rank)
                        
                        mrr += 1/(rank+1)
                        filtered_mrr += 1/(filtered_rank+1)
                    
                        if rank == 0:
                            hits_at_1 += 1
                        if rank < 3:
                            hits_at_3 += 1
                        if rank < 10:
                            hits_at_10 += 1
                        if rank < 100:
                            hits_at_100 += 1

                        if filtered_rank == 0:
                            fhits_at_1 += 1
                        if filtered_rank < 3:
                            fhits_at_3 += 1
                        if filtered_rank < 10:
                            fhits_at_10 += 1
                        if filtered_rank < 100:
                            fhits_at_100 += 1

                        if filtered_rank not in filtered_ranks:
                            filtered_ranks[filtered_rank] = 0
                        filtered_ranks[filtered_rank] += 1

            mean_rank /= eval_dl.dataset_len
            if "test" in mode:
                mrr /= eval_dl.dataset_len
                hits_at_1 /= eval_dl.dataset_len
                hits_at_3 /= eval_dl.dataset_len
                hits_at_10 /= eval_dl.dataset_len
                hits_at_100 /= eval_dl.dataset_len
                auc = self.compute_rank_roc(ranks)
                median_rank = np.median(rank_vals)
                percentile_90 = np.percentile(rank_vals, 90)
                below_1000 = self.calculate_percentile_1000(ranks)
                
                filtered_mean_rank /= eval_dl.dataset_len
                filtered_mrr /= eval_dl.dataset_len
                fhits_at_1 /= eval_dl.dataset_len
                fhits_at_3 /= eval_dl.dataset_len
                fhits_at_10 /= eval_dl.dataset_len
                fhits_at_100 /= eval_dl.dataset_len
                fauc = self.compute_rank_roc(filtered_ranks)
                fmedian_rank = np.median(filtered_rank_vals)
                fpercentile_90 = np.percentile(filtered_rank_vals, 90)
                fbelow_1000 = self.calculate_percentile_1000(filtered_ranks)
                
                raw_metrics = (mean_rank, mrr, median_rank, hits_at_1, hits_at_3, hits_at_10, hits_at_100, auc, percentile_90, below_1000)
                filtered_metrics = (filtered_mean_rank, filtered_mrr, fmedian_rank, fhits_at_1, fhits_at_3, fhits_at_10, fhits_at_100, fauc, fpercentile_90, fbelow_1000)
        if "test" in mode:
            return raw_metrics, filtered_metrics
        else:
            return mean_rank
                                                                                                

    def predict(self, heads, tails, mode, model=None):
        
        aux = heads.to(self.device)
        num_heads = len(heads)

        tail_ids = th.arange(len(self.dataset.classes)).to(self.device)
                                                            
        heads = heads.to(self.device)
        heads = heads.repeat(len(tail_ids), 1).T
        assert (heads[0,:] == aux[0]).all(), f"{heads[0,:]}, {aux[0]}"
        heads = heads.reshape(-1)
        assert (aux[0] == heads[:len(tail_ids)]).all(), "heads are not the same"
        assert (aux[1] == heads[len(tail_ids): 2*len(tail_ids)]).all(), "heads are not the same"
                        
        eval_tails = tail_ids.repeat(num_heads)
        assert (eval_tails.shape == heads.shape), f"eval_tails shape: {eval_tails.shape}, heads shape: {heads.shape}"

        data = th.stack((heads, eval_tails), dim=1)
        assert data.shape[1] == 2, f"Data shape: {data.shape}"
        

        if "test" in mode:
            if self.aggregator == "mean":
                aggregator = th.mean
            elif self.aggregator == "max":
                aggregator = lambda *args, **kwargs: th.max(*args, **kwargs).values
            elif self.aggregator == "min":
                aggregator = lambda *args, **kwargs: th.min(*args, **kwargs).values
            elif self.aggregator == "median":
                aggregator = lambda *args, **kwargs: th.median(*args, **kwargs).values
                
            predictions = []
            for module in self.modules:
                module.eval()
                module.to(self.device)

                if "subsumption" in mode:    
                    curr_preds = -module.tbox_forward(data, "gci0")
                elif "membership" in mode:
                    curr_preds = module.abox_forward(aux)
                    max_ = th.max(curr_preds)
                    curr_preds = curr_preds - max_

                else:
                    raise ValueError(f"Mode {mode} not recognized")
                predictions.append(curr_preds)
            predictions = th.stack(predictions, dim=1)
            predictions = aggregator(predictions, dim=1)
            predictions = predictions.reshape(-1, len(tail_ids))

        else:
            predictions = []
            module = self.modules[model]
            module.eval()
            module.to(self.device)
            if "subsumption" in mode:
                predictions = -module.tbox_forward(data, "gci0")
            elif "membership" in mode:
                predictions = module.abox_forward(aux)
                max_ = th.max(predictions)
                predictions = predictions - max_
            predictions = predictions.reshape(-1, len(tail_ids))
                
        return predictions
        



    def test(self):
        logging.info("Testing...")
        filtering_labels = self.get_filtering_labels()
        subsumption_metrics = self.compute_ranking_metrics(filtering_labels[0], "test_subsumption")
        membership_metrics = self.compute_ranking_metrics(filtering_labels[1], "test_membership")
        return subsumption_metrics, membership_metrics


    def load_best_model(self):
        for i, module in enumerate(self.modules):
            logger.info(f"Loading model {i+1} of {len(self.modules)}")
            sub_module_filepath = self.model_filepath.replace(".pt", f"_{i+1}_of_{len(self.modules)}.pt")
            module.load_state_dict(th.load(sub_module_filepath))
            
            
            
    def compute_rank_roc(self, ranks):
        n_tails = len(self.dataset.classes)
                    
        auc_x = list(ranks.keys())
        auc_x.sort()
        auc_y = []
        tpr = 0
        sum_rank = sum(ranks.values())
        for x in auc_x:
            tpr += ranks[x]
            auc_y.append(tpr / sum_rank)
        auc_x.append(n_tails)
        auc_y.append(1)
        auc = np.trapz(auc_y, auc_x) / n_tails
        return auc

    def calculate_percentile_1000(self, scores):
        ranks_1000=[]
        for item in scores:
            if item < 1000:
                ranks_1000.append(item)
        n_1000 = len(ranks_1000)
        nt = len(scores)
        percentile = (n_1000/nt)*100
        return percentile
