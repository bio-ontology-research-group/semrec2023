import mowl
mowl.init_jvm("10g")
from mowl.datasets import PathDataset
from mowl.owlapi import OWLAPIAdapter

from org.semanticweb.owlapi.formats import OWLXMLDocumentFormat
from org.semanticweb.owlapi.model import IRI
from java.util import HashSet

import os
import sys


owl2bench_number = int(sys.argv[1])
root = f"../use_cases/owl2bench{owl2bench_number}/data"
if owl2bench_number < 1 or owl2bench_number > 3:
    raise ValueError("owl2bench_number must be 1, or 2")

owl2bench_owl = f"{root}/OWL2DL-{owl2bench_number}.owl"
train_owl = f"{root}/_train_OWL2Bench{owl2bench_number}_wrapped.owl"
valid_owl = f"{root}/_valid_OWL2Bench{owl2bench_number}_wrapped.owl"
test_owl = f"{root}/_test_OWL2Bench{owl2bench_number}_wrapped.owl"

output_file = f"{root}/OWL2DL-{owl2bench_number}_cleaned.owl"

# Get validation and testing axioms
valid_dataset = PathDataset(valid_owl)
test_dataset = PathDataset(test_owl)
valid_axioms = set(valid_dataset.ontology.getAxioms())
test_axioms = set(test_dataset.ontology.getAxioms())

# Get training axioms
owl2bench_dataset = PathDataset(owl2bench_owl)
train_dataset = PathDataset(train_owl)
owl2bench_axioms = set(owl2bench_dataset.ontology.getAxioms())
train_axioms = set(train_dataset.ontology.getAxioms())

# Remove validation and testing axioms from training axioms
full_train_axioms = owl2bench_axioms.union(train_axioms)
full_train_axioms = full_train_axioms.difference(valid_axioms)
full_train_axioms = full_train_axioms.difference(test_axioms)

new_axioms_set = HashSet()
new_axioms_set.addAll(full_train_axioms)

# Save training axioms to file
adapter = OWLAPIAdapter()
manager = adapter.owl_manager
new_ontology = manager.createOntology()
new_ontology.addAxioms(new_axioms_set)
manager.saveOntology(new_ontology, OWLXMLDocumentFormat(), IRI.create("file:" + os.path.abspath(output_file)))

