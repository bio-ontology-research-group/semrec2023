import random
import numpy as np
import torch as th
import os
from org.semanticweb.owlapi.model import AxiomType as ax
from itertools import chain, combinations, product


bot_name = {
    "owl2vec": "http://www.w3.org/2002/07/owl#Nothing",
    "onto2graph": "http://www.w3.org/2002/07/owl#Nothing",
    "rdf": "http://www.w3.org/2002/07/owl#Nothing",
    "cat": "owl:Nothing",
    "cat1": "owl:Nothing",
    "cat2": "owl:Nothing", 
}

top_name = {
    "owl2vec": "http://www.w3.org/2002/07/owl#Thing",
    "onto2graph": "http://www.w3.org/2002/07/owl#Thing",
    "rdf": "http://www.w3.org/2002/07/owl#Thing",
    "cat": "owl:Thing",
    "cat1": "owl:Thing",
    "cat2": "owl:Thing",
    }



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False



IGNORED_AXIOM_TYPES = [ax.ANNOTATION_ASSERTION,
                       ax.ASYMMETRIC_OBJECT_PROPERTY,
                       ax.DECLARATION,
                       ax.EQUIVALENT_OBJECT_PROPERTIES,
                       ax.FUNCTIONAL_OBJECT_PROPERTY,
                       ax.INVERSE_FUNCTIONAL_OBJECT_PROPERTY,
                       ax.INVERSE_OBJECT_PROPERTIES,
                       ax.IRREFLEXIVE_OBJECT_PROPERTY,
                       ax.OBJECT_PROPERTY_DOMAIN,
                       ax.OBJECT_PROPERTY_RANGE,
                       ax.REFLEXIVE_OBJECT_PROPERTY,
                       ax.SUB_PROPERTY_CHAIN_OF,
                       ax.SUB_ANNOTATION_PROPERTY_OF,
                       ax.SUB_OBJECT_PROPERTY,
                       ax.SWRL_RULE,
                       ax.SYMMETRIC_OBJECT_PROPERTY,
                       ax.TRANSITIVE_OBJECT_PROPERTY
                       ]

def pairs(iterable):
    num_items = len(iterable)
    power_set = list(powerset(iterable))
    product_set = list(product(power_set, power_set))

    curated_set = []
    for i1, i2 in product_set:
        if i1 == i2:
            continue
        if len(i1) + len(i2) != num_items:
            continue
        if len(i1) == 0 or len(i1) == num_items:
            continue
        if len(i2) == 0 or len(i2) == num_items:
            continue
        curated_set.append((i1, i2))

    return curated_set
