import torch.nn as nn
import os
import pandas as pd
from pykeen.models import TransE, DistMult, ConvKB, ERModel
from mowl.owlapi.defaults import TOP, BOT
from mowl.projection import OWL2VecStarProjector
from mowl.datasets import PathDataset

from .cat_projector import CategoricalProjector

import torch as th
from mowl.utils.data import FastTensorDataLoader
from src.utils import bot_name, top_name
from src.nn import KGEModule
from pykeen.triples import TriplesFactory
from tqdm import tqdm
import numpy as np
import pickle as pkl

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GraphModel():
    def __init__(self,
                 use_case,
                 graph_type,
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
                 initial_tolerance,
                 ):

        self.use_case = use_case
        self.graph_type = graph_type
        self.kge_model = kge_model
        self.root = root
        self.emb_dim = emb_dim
        self.margin = margin
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lr = lr
        self.num_negs = num_negs
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.device = device
        self.seed = seed
        self.initial_tolerance = initial_tolerance
                
        self._triples_factory = None
        
        self._train_graph = None
        self._valid_subsumption_graph = None
        self._valid_membership_graph = None
        self._test_subsumption_graph = None
        self._test_membership_graph = None
        
        self._model_path = None
        self._node_to_id = None
        self._relation_to_id = None
        self._id_to_node = None
        self._id_to_relation = None
        self._classes = None
        self._object_properties = None
        self._individuals = None
        self._classes_ids = None
        self._object_properties_ids = None
        self._individuals_ids = None
                                
        print("Parameters:")
        print(f"\tUse case: {self.use_case}")
        print(f"\tGraph type: {self.graph_type}")
        print(f"\tKGE model: {self.kge_model}")
        print(f"\tRoot: {self.root}")
        print(f"\tEmbedding dimension: {self.emb_dim}")
        print(f"\tMargin: {self.margin}")
        print(f"\tWeight decay: {self.weight_decay}")
        print(f"\tBatch size: {self.batch_size}")
        print(f"\tLearning rate: {self.lr}")
        print(f"\tNumber of negatives: {self.num_negs}")
        print(f"\tTest batch size: {self.test_batch_size}")
        print(f"\tEpochs: {self.epochs}")
        print(f"\tDevice: {self.device}")
        print(f"\tSeed: {self.seed}")
                

        if self.graph_type == "owl2vec":
            self.projector = OWL2VecStarProjector(bidirectional_taxonomy=True)
        elif self.graph_type == "cat":
            self.projector = CategoricalProjector()
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
        
        if "ore" in use_case:
            number = use_case[-1]
            self.train_path = os.path.join(self.root, f"ORE{number}_cleaned.owl") # given and modified a priori
            self.valid_path = os.path.join(self.root, f"_valid_ORE{number}_wrapped.owl") # given
            self.test_path = os.path.join(self.root, f"_test_ORE{number}_wrapped.owl") # given
            self._train_graph_path = os.path.join(self.root, f"ORE{number}_{graph_type}.edgelist") # constructed a priori
            self._valid_subsumption_graph_path = os.path.join(self.root, f"ORE{number}_subsumption_valid.edgelist")
            self._test_subsumption_graph_path = os.path.join(self.root, f"ORE{number}_subsumption_test.edgelist") # constructed on the fly and saved
            self._valid_membership_graph_path = os.path.join(self.root, f"ORE{number}_membership_valid.edgelist")
            self._test_membership_graph_path = os.path.join(self.root, f"ORE{number}_membership_test.edgelist") # constructed on the fly and saved
        elif "owl2bench" in use_case:
            number = use_case[-1]
            self.train_path = os.path.join(self.root, f"OWL2DL-{number}_cleaned.owl") # given and modified a priori
            self.valid_path = os.path.join(self.root, f"_valid_OWL2Bench{number}_wrapped.owl") # given
            self.test_path = os.path.join(self.root, f"_test_OWL2Bench{number}_wrapped.owl") # given
            self._train_graph_path = os.path.join(self.root, f"OWL2DL-{number}_{graph_type}.edgelist") # constructed a priori
            self._valid_subsumption_graph_path = os.path.join(self.root, f"OWL2DL-{number}_subsumption_valid.edgelist")
            self._test_subsumption_graph_path = os.path.join(self.root, f"OWL2DL-{number}_subsumption_test.edgelist") # constructed on the fly and saved
            self._valid_membership_graph_path = os.path.join(self.root, f"OWL2DL-{number}_membership_valid.edgelist")
            self._test_membership_graph_path = os.path.join(self.root, f"OWL2DL-{number}_membership_test.edgelist") # constructed on the fly and saved
        elif "caligraph" in use_case:
            number = use_case[-1]
            self.train_path = os.path.join(self.root, f"clg_10e{number}_cleaned.owl") # given and modified a priori
            self.valid_path = os.path.join(self.root, f"clg_10e{number}-val_corrected.owl")
            self.test_path = os.path.join(self.root, f"clg_10e{number}-test_corrected.owl")
            self._train_graph_path = os.path.join(self.root, f"clg_10e{number}_{graph_type}.edgelist") # constructed a priori
            self._valid_subsumption_graph_path = os.path.join(self.root, f"clg_10e{number}_subsumption_valid.edgelist")
            self._test_subsumption_graph_path = os.path.join(self.root, f"clg_10e{number}_subsumption_test.edgelist") # constructed on the fly and saved
            self._valid_membership_graph_path = os.path.join(self.root, f"clg_10e{number}_membership_valid.edgelist")
            self._test_membership_graph_path = os.path.join(self.root, f"clg_10e{number}_membership_test.edgelist") # constructed on the fly and saved
            
        else:
            raise ValueError(f"Unknown use case: {use_case}")

        self._classes_path = os.path.join(self.root, f"classes.tsv") # constructed on the fly and saved
        self._object_properties_path = os.path.join(self.root, f"properties.tsv") # constructed on the fly and saved
        self._individuals_path = os.path.join(self.root, f"individuals.tsv") # constructed on the fly and saved
        
        self.dataset = PathDataset(self.train_path, validation_path = self.valid_path, testing_path = self.test_path)
        
        self.model = KGEModule(kge_model,
                               triples_factory=self.triples_factory,
                               embedding_dim=self.emb_dim,
                               random_seed=self.seed)


    def _load_graph(self, path, mode="train"):
        if os.path.exists(path):
            logger.info(f"Loading graph from {path}")
            graph = pd.read_csv(path, sep="\t", header=None)
            graph.columns = ["head", "relation", "tail"]
            if self.graph_type == "cat":
                graph["relation"] = graph["relation"].replace({"http://subclassof": "http://arrow"})
                
        else:
            logger.info(f"Graph {path} does not exist. Generating it...")
            if mode == "train":
                edges = self.projector.project(self.dataset.ontology)
            elif "valid" in mode:
                edges = self.projector.project(self.dataset.validation)
            elif "test" in mode:
                edges = self.projector.project(self.dataset.testing)
                
            edges = [(e.src, e.rel, e.dst) for e in edges]
            graph = pd.DataFrame(edges, columns=["head", "relation", "tail"])
                

            if mode == "train":
                graph.to_csv(path, sep="\t", header=None, index=False)
            elif "subsumption" in mode:
                graph = graph[graph["relation"] == "http://subclassof"]
                                    
                graph.to_csv(path, sep="\t", header=None, index=False)
            elif "membership" in mode:
                graph = graph[graph["relation"] == "http://type"]
                graph.to_csv(path, sep="\t", header=None, index=False)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            graph = pd.read_csv(path, sep="\t", header=None)
            graph.columns = ["head", "relation", "tail"]
          
            if self.graph_type == "cat":
                graph["relation"] = graph["relation"].replace({"http://subclassof": "http://arrow"})
                
        logger.info(f"Loaded {mode} graph with {len(graph)} edges")
        return graph

    
            
    @property
    def train_graph(self):
        if self._train_graph is not None:
            return self._train_graph

        self._train_graph = self._load_graph(self._train_graph_path, mode="train")
        return self._train_graph

    @property
    def valid_subsumption_graph(self):
        if self._valid_subsumption_graph is not None:
            return self._valid_subsumption_graph

        self._valid_subsumption_graph = self._load_graph(self._valid_subsumption_graph_path, mode="valid_subsumption")
        return self._valid_subsumption_graph
                                                            
    @property
    def valid_membership_graph(self):
        if self._valid_membership_graph is not None:
            return self._valid_membership_graph

        self._valid_membership_graph = self._load_graph(self._valid_membership_graph_path, mode="valid_membership")
        return self._valid_membership_graph

    @property
    def test_subsumption_graph(self):
        if self._test_subsumption_graph is not None:
            return self._test_subsumption_graph

        self._test_subsumption_graph = self._load_graph(self._test_subsumption_graph_path, mode="test_subsumption")
        return self._test_subsumption_graph

    @property
    def test_membership_graph(self):
        if self._test_membership_graph is not None:
            return self._test_membership_graph

        self._test_membership_graph = self._load_graph(self._test_membership_graph_path, mode="test_membership")
        return self._test_membership_graph
    
    @property
    def classes(self):
        if self._classes is not None:
            return self._classes

        if os.path.exists(self._classes_path):
            logger.info(f"Loading classes from {self._classes_path}")
            classes = pd.read_csv(self._classes_path, sep="\t", header=None)
            classes.columns = ["class"]
            classes = list(classes["class"].values.flatten())
            classes.sort()
            self._classes = classes
        else:
            logger.info(f"Classes  do not exist. Generating it...")
            classes = set(self.dataset.ontology.getClassesInSignature())
            classes |= set(self.dataset.validation.getClassesInSignature())
            classes |= set(self.dataset.testing.getClassesInSignature())
            classes = sorted(list(classes))
            classes = [str(c.toStringID()) for c in classes]
            classes = pd.DataFrame(classes, columns=["class"])
            classes.to_csv(self._classes_path, sep="\t", header=None, index=False)
            classes = list(classes["class"].values.flatten())
            classes.sort()
            self._classes = classes
         
        return self._classes

    @property
    def object_properties(self):
        if self._object_properties is not None:
            return self._object_properties

        if os.path.exists(self._object_properties_path):
            logger.info(f"Loading properties from {self._object_properties_path}")
            properties = pd.read_csv(self._object_properties_path, sep="\t", header=None)
            properties.columns = ["property"]
            properties = list(properties["property"].values.flatten())
            properties.sort()
            self._object_properties = properties
            
        else:
            logger.info(f"Properties do not exist. Generating it...")
            properties = set(self.dataset.ontology.getObjectPropertiesInSignature())
            properties |= set(self.dataset.validation.getObjectPropertiesInSignature())
            properties |= set(self.dataset.testing.getObjectPropertiesInSignature())
            properties = sorted(list(properties))
            properties = [str(p.toStringID()) for p in properties]
            properties = pd.DataFrame(properties, columns=["property"])
            properties.to_csv(self._object_properties_path, sep="\t", header=None, index=False)
            properties = list(properties["property"].values.flatten())
            properties.sort()
            self._object_properties = properties
            
        return self._object_properties

    @property
    def individuals(self):
        if self._individuals is not None:
            return self._individuals

        if os.path.exists(self._individuals_path):
            logger.info(f"Loading individuals from {self._individuals_path}")
            individuals = pd.read_csv(self._individuals_path, sep="\t", header=None)
            individuals.columns = ["individual"]
            individuals = list(individuals["individual"].values.flatten())
            individuals.sort()
            self._individuals = individuals
            
        else:
            logger.info(f"Individuals do not exist. Generating it...")
            individuals = set(self.dataset.ontology.getIndividualsInSignature())
            individuals |= set(self.dataset.validation.getIndividualsInSignature())
            individuals |= set(self.dataset.testing.getIndividualsInSignature())
            individuals = sorted(list(individuals))
            individuals = [str(i.toStringID()) for i in individuals]
            individuals = pd.DataFrame(individuals, columns=["individual"])
            individuals.to_csv(self._individuals_path, sep="\t", header=None, index=False)
            individuals = list(individuals["individual"].values.flatten())
            individuals.sort()
            self._individuals = individuals
        return self._individuals
    
    @property
    def model_path(self):
        if self._model_path is not None:
            return self._model_path

        params_str = f"{self.graph_type}"
        params_str += f"_kge_{self.kge_model}"
        #params_str += f"_dim{self.emb_dim}"
        #params_str += f"_marg{self.margin}"
        #params_str += f"_reg{self.weight_decay}"
        #params_str += f"_bs{self.batch_size}"
        #params_str += f"_lr{self.lr}"
        #params_str += f"_negs{self.num_negs}"
        
        models_dir = os.path.dirname(self.root)
        models_dir = os.path.join(models_dir, "models")

        basename = f"{params_str}.model.pt"
        self._model_path = os.path.join(models_dir, basename)
        return self._model_path

    @property
    def node_to_id(self):
        if self._node_to_id is not None:
            return self._node_to_id

        graph_classes = set(self.train_graph["head"].unique()) | set(self.train_graph["tail"].unique())
        graph_classes |= set(self.valid_subsumption_graph["head"].unique()) | set(self.valid_subsumption_graph["tail"].unique())
        graph_classes |= set(self.valid_membership_graph["head"].unique()) | set(self.valid_membership_graph["tail"].unique())
        graph_classes |= set(self.test_subsumption_graph["head"].unique()) | set(self.test_subsumption_graph["tail"].unique())
        graph_classes |= set(self.test_membership_graph["head"].unique()) | set(self.test_membership_graph["tail"].unique())
        
        bot = bot_name[self.graph_type]
        top = top_name[self.graph_type]
        graph_classes.add(bot)
        graph_classes.add(top)
                
        ont_classes = set(self.classes)
        all_classes = list(graph_classes | ont_classes | set(self.individuals)) 
        all_classes.sort()
        self._node_to_id = {c: i for i, c in enumerate(all_classes)}
        logger.info(f"Number of graph nodes: {len(self._node_to_id)}")
        return self._node_to_id
    
    @property
    def id_to_node(self):
        if self._id_to_node is not None:
            return self._id_to_node
        
        id_to_node =  {v: k for k, v in self.node_to_id.items()}
        self._id_to_node = id_to_node
        return self._id_to_node
    
    @property
    def relation_to_id(self):
        if self._relation_to_id is not None:
            return self._relation_to_id

        graph_rels = list(self.train_graph["relation"].unique())
        graph_rels.sort()
        self._relation_to_id = {r: i for i, r in enumerate(graph_rels)}
        logger.info(f"Number of graph relations: {len(self._relation_to_id)}")
        return self._relation_to_id

    @property
    def id_to_relation(self):
        if self._id_to_relation is not None:
            return self._id_to_relation

        id_to_relation = {v: k for k, v in self.relation_to_id.items()}
        self._id_to_relation = id_to_relation
        return self._id_to_relation

    
    @property
    def triples_factory(self):
        if self._triples_factory is not None:
            return self._triples_factory

        tensor = []
        for row in self.train_graph.itertuples():
            tensor.append([self.node_to_id[row.head],
                           self.relation_to_id[row.relation],
                           self.node_to_id[row.tail]])

        tensor = th.LongTensor(tensor)
        self._triples_factory = TriplesFactory(tensor, self.node_to_id, self.relation_to_id, create_inverse_triples=True)
        return self._triples_factory


    @property
    def classes_ids(self):
        """
        This is the class to id mapping for the ontology classes
        The indices are taken from the graph

        :rtype: torch.Tensor
        """
        if self._classes_ids is not None:
            return self._classes_ids
        
        class_to_id = {c: self.node_to_id[c] for c in self.classes}
        ontology_classes_idxs = th.tensor(list(class_to_id.values()), dtype=th.long, device=self.device)
        self._classes_ids = ontology_classes_idxs
        return self._classes_ids

    @property
    def object_properties_ids(self):
        """
        This is the object property to id mapping for the ontology object properties
        The indices are taken from the graph

        :rtype: torch.Tensor
        """
        if self._object_property_ids is not None:
            return self._object_property_ids
        
        if self.graph_type in ["cat"]:
            prop_to_id = {c: self.node_to_id[c] for c in self.object_properties if c in self.node_to_id}
        else:
            prop_to_id = {c: self.relation_to_id[c] for c in self.object_properties if c in self.relation_to_id}
        
        object_property_to_id = th.tensor(list(prop_to_id.values()), dtype=th.long, device=self.device)
        self._object_properties_ids = object_properties_ids
        return self._object_properties_ids

    @property
    def individuals_ids(self): 
        """
        This is the individual to id mapping for the ontology individuals
        The indices are taken from the graph

        :rtype: torch.Tensor
        """
        if self._individuals_ids is not None:
            return self._individuals_ids

        individual_to_id = {c: self.node_to_id[c] for c in self.individuals}
        individual_to_id = th.tensor(list(individual_to_id.values()), dtype=th.long, device=self.device)
        self._individuals_ids = individual_to_id
        return self._individuals_ids

    def create_graph_dataloader(self, mode="train", batch_size=None):
        if batch_size is None:
            raise ValueError("Batch size must be specified")
        
        if mode == "train":
            graph = self.train_graph
        elif mode == "valid_subsumption":
            graph = self.valid_subsumption_graph
        elif mode == "valid_membership":
            graph = self.valid_membership_graph
        elif mode == "test_subsumption":
            graph = self.test_subsumption_graph
        elif mode == "test_membership":
            graph = self.test_membership_graph
        else:
            raise ValueError(f"Unknown mode: {mode}")

        
        heads = [self.node_to_id[h] for h in graph["head"]]
        rels = [self.relation_to_id[r] for r in graph["relation"]]
        tails = [self.node_to_id[t] for t in graph["tail"]]

        heads = th.LongTensor(heads)
        rels = th.LongTensor(rels)
        tails = th.LongTensor(tails)
        
        dataloader = FastTensorDataLoader(heads, rels, tails,
                                          batch_size=batch_size, shuffle=True)
        return dataloader
    

    
    def train(self):
        raise NotImplementedError


            
    def get_filtering_labels(self):
        logger.info("Getting predictions and labels")

        num_test_subsumption_heads = len(self.classes)
        num_test_subsumption_tails = len(self.classes)
        num_test_membership_heads = len(self.individuals)
        num_test_membership_tails = len(self.classes)

        if self.graph_type == "owl2vec":
            subsumption_relation = "http://subclassof"
            membership_relation = "http://type"
        elif self.graph_type == "cat":
            subsumption_relation = "http://arrow"
            membership_relation = "http://type"
            
        # get predictions and labels for test subsumption
        filter_subsumption_labels = np.ones((num_test_subsumption_heads, num_test_subsumption_tails), dtype=np.int32)
        filter_membership_labels = np.ones((num_test_membership_heads, num_test_membership_tails), dtype=np.int32)
        
        classes_ids = self.classes_ids.to(self.device)
        individuals_ids = self.individuals_ids.to(self.device)
        
        train_dataloader = self.create_graph_dataloader(mode="train", batch_size=self.batch_size)
        subsumption_triples = 0
        membership_triples = 0
        with th.no_grad():
            for head_idxs, rel_idxs, tail_idxs in tqdm(train_dataloader, desc="Getting labels"):
                head_idxs = head_idxs.to(self.device)
                
                for i, head_graph_id in enumerate(head_idxs):
                    rel = rel_idxs[i]
                    rel_name = self.id_to_relation[rel.item()]
                    if rel_name == subsumption_relation:
                        tail_graph_id = tail_idxs[i]
                        
                        head_class = self.id_to_node[head_graph_id.item()]
                        tail_class = self.id_to_node[tail_graph_id.item()]
                        if not head_class in self.classes or not tail_class in self.classes:
                            continue
                                                
                        head_ont_id = th.where(self.classes_ids == head_graph_id)[0]
                        rel = rel_idxs[i]
                        
                        tail_ont_id = th.where(self.classes_ids == tail_graph_id)[0]
                        filter_subsumption_labels[head_ont_id, tail_ont_id] = 10000
                        subsumption_triples += 1
                    elif rel_name == membership_relation:
                        tail_graph_id = tail_idxs[i]

                        head_individual = self.id_to_node[head_graph_id.item()]
                        tail_class = self.id_to_node[tail_graph_id.item()]
                        if not head_individual in self.individuals or not tail_class in self.classes:
                            continue
                        head_ont_id = th.where(self.individuals_ids == head_graph_id)[0]
                        
                        tail_ont_id = th.where(self.classes_ids == tail_graph_id)[0]
                        filter_membership_labels[head_ont_id, tail_ont_id] = 10000
                        membership_triples += 1
                    else:
                        continue

        logger.info(f"Subsumption filtering triples: {subsumption_triples}")
        logger.info(f"Membership filtering triples: {membership_triples}")
        return filter_subsumption_labels, filter_membership_labels

    def load_best_model(self):
        logger.info(f"Loading best model from {self.model_path}")
        self.model.load_state_dict(th.load(self.model_path))
        self.model = self.model.to(self.device)

    def compute_ranking_metrics(self, filtering_labels=None, mode="test_subsumption"):
        if not mode in ["test_subsumption", "valid_subsumption", "test_membership", "valid_membership"]:
            raise ValueError(f"Invalid mode {mode}")

        if filtering_labels is None and "test" in mode:
            raise ValueError("filtering_labels cannot be None when mode is test")

        if filtering_labels is not None and "validate" in mode:
            raise ValueError("filtering_labels must be None when mode is validate")


        if "test" in mode:
            self.load_best_model()

        all_tail_ids = self.classes_ids.to(self.device)
        if "subsumption" in mode:
            all_head_ids = self.classes_ids.to(self.device)
        elif "membership" in mode:
            all_head_ids = self.individuals_ids.to(self.device)
        else:
            raise ValueError(f"Invalid mode {mode}")
        
        self.model.eval()
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

        dataloader = self.create_graph_dataloader(mode, batch_size=self.test_batch_size)
        with th.no_grad():
            for head_idxs, rel_idxs, tail_idxs in tqdm(dataloader, desc="Computing metrics..."):

                predictions = self.predict(head_idxs, rel_idxs, tail_idxs, mode)
                
                assert predictions.shape[0] == head_idxs.shape[0], f"Predictions shape: {predictions.shape}, head_idxs shape: {head_idxs.shape}"
                
                for i, graph_head in enumerate(head_idxs):
                    graph_tail = tail_idxs[i]
                    head = th.where(all_head_ids == graph_head)[0]
                    tail = th.where(all_tail_ids == graph_tail)[0]
                    
                    logger.debug(f"graph_tail: {graph_tail}")
                    
                    rel = rel_idxs[i]
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

            mean_rank /= dataloader.dataset_len
            if "test" in mode:
                mrr /= dataloader.dataset_len
                hits_at_1 /= dataloader.dataset_len
                hits_at_3 /= dataloader.dataset_len
                hits_at_10 /= dataloader.dataset_len
                hits_at_100 /= dataloader.dataset_len
                auc = self.compute_rank_roc(ranks)
                median_rank = np.median(rank_vals)
                percentile90 = np.percentile(rank_vals, 90)
                below_1000 = self.calculate_percentile_1000(ranks)
                
                filtered_mean_rank /= dataloader.dataset_len
                filtered_mrr /= dataloader.dataset_len
                fhits_at_1 /= dataloader.dataset_len
                fhits_at_3 /= dataloader.dataset_len
                fhits_at_10 /= dataloader.dataset_len
                fhits_at_100 /= dataloader.dataset_len
                fauc = self.compute_rank_roc(filtered_ranks)
                fmedian_rank = np.median(filtered_rank_vals)
                fpercentile90 = np.percentile(filtered_rank_vals, 90)
                fbelow_1000 = self.calculate_percentile_1000(filtered_ranks)
                
                raw_metrics = (mean_rank, mrr, median_rank, hits_at_1, hits_at_3, hits_at_10, hits_at_100, auc, percentile90, below_1000)
                filtered_metrics = (filtered_mean_rank, filtered_mrr, fmedian_rank, fhits_at_1, fhits_at_3, fhits_at_10, fhits_at_100, fauc, fpercentile90, fbelow_1000)
        if "test" in mode:
            return raw_metrics, filtered_metrics
        else:
            return mean_rank
        
                                                                                     
    def normal_forward(self, head_idxs, rel_idxs, tail_idxs, n_classes):
        logits = self.model.predict((head_idxs, rel_idxs, tail_idxs))
        logger.debug(f"logits shape before reshape: {logits.shape}")
        logits = logits.reshape(-1, n_classes)
        logger.debug(f"logits shape after reshape: {logits.shape}")
        return logits

    def predict(self, heads, rels, tails, mode):

        aux = heads.to(self.device)
        num_heads = len(heads)

        tail_ids = self.classes_ids.to(self.device)
                                                            
        heads = heads.to(self.device)
        heads = heads.repeat(len(tail_ids), 1).T
        assert (heads[0,:] == aux[0]).all(), f"{heads[0,:]}, {aux[0]}"
        heads = heads.reshape(-1)
        assert (aux[0] == heads[:len(tail_ids)]).all(), "heads are not the same"
        assert (aux[1] == heads[len(tail_ids): 2*len(tail_ids)]).all(), "heads are not the same"
        rels = rels.to(self.device)
        rels = rels.repeat(len(tail_ids),1).T
        rels = rels.reshape(-1)

        eval_tails = tail_ids.repeat(num_heads)
        assert (eval_tails.shape == rels.shape == heads.shape), f"eval_tails shape: {eval_tails.shape}, rels shape: {rels.shape}, heads shape: {heads.shape}"
        logits = self.normal_forward(heads, rels, eval_tails, len(tail_ids))

        return logits
        


    
    def test(self):
        logger.info("Testing ontology completion...")
        filtering_labels = self.get_filtering_labels()
        subsumption_metrics = self.compute_ranking_metrics(filtering_labels[0], "test_subsumption")
        membership_metrics = self.compute_ranking_metrics(filtering_labels[1], "test_membership")
        return subsumption_metrics, membership_metrics

    def compute_rank_roc(self, ranks):
        n_tails = len(self.classes_ids)
                    
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
             


    def save_embeddings_data(self):
        out_class_file = os.path.join(self.root, f"{self.use_case}.{self.graph_type}.class_embeddings.pkl")
        out_individual_file = os.path.join(self.root, f"{self.use_case}.{self.graph_type}.individual_embeddings.pkl")
        out_role_file = os.path.join(self.root, f"{self.use_case}.{self.graph_type}.role_embeddings.pkl")
        out_triples_factory_file = os.path.join(self.root, f"triples_factory.pkl")
        
        cls_ids = [self.node_to_id[n] for n in self.classes]
        ind_ids = [self.node_to_id[n] for n in self.individuals]
        if self.graph_type == "owl2vec":
            role_ids = [self.relation_to_id[n] for n in ["http://subclassof", "http://type"]]
        elif self.graph_type == "cat":
            role_ids = [self.relation_to_id[n] for n in ["http://arrow", "http://type"]]
            
        cls_df = pd.DataFrame(list(zip(self.classes, cls_ids)), columns=["class", "node_id"])
        inds_df = pd.DataFrame(list(zip(self.individuals, ind_ids)) , columns=["individual", "node_id"])
        role_df = pd.DataFrame(list(zip(["http://subclassof", "http://type"], role_ids)), columns=["role", "relation_id"])
        
        cls_df.to_pickle(out_class_file)
        logger.info(f"Saved class data to {out_class_file}")
        inds_df.to_pickle(out_individual_file)
        logger.info(f"Saved individual data to {out_individual_file}")
        role_df.to_pickle(out_role_file)
        logger.info(f"Saved role data to {out_role_file}")
        
        pkl.dump(self.triples_factory, open(out_triples_factory_file, "wb"))
        logger.info(f"Saved triples factory to {out_triples_factory_file}")

    def calculate_percentile_1000(self, scores):
        ranks_1000=[]
        for item in scores:
            if item < 1000:
                ranks_1000.append(item)
        n_1000 = len(ranks_1000)
        nt = len(scores)
        percentile = (n_1000/nt)*100
        return percentile
