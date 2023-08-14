import mowl
mowl.init_jvm("10g")
from mowl.datasets import PathDataset
from mowl.owlapi import OWLAPIAdapter

from org.semanticweb.owlapi.formats import OWLXMLDocumentFormat
from org.semanticweb.owlapi.model import IRI
from java.util import HashSet

import os
import sys


ore_number = int(sys.argv[1])
root = f"../use_cases/ore{ore_number}/data"
if ore_number < 1 or ore_number > 3:
    raise ValueError("ore_number must be 1, 2 or 3")

ore_owl = f"{root}/ORE{ore_number}.owl"
train_owl = f"{root}/_train_ORE{ore_number}_wrapped.owl"
valid_owl = f"{root}/_valid_ORE{ore_number}_wrapped.owl"
test_owl = f"{root}/_test_ORE{ore_number}_wrapped.owl"

output_file = f"{root}/ORE{ore_number}_cleaned.owl"

# Get validation and testing axioms
valid_dataset = PathDataset(valid_owl)
test_dataset = PathDataset(test_owl)
valid_axioms = set(valid_dataset.ontology.getAxioms())
test_axioms = set(test_dataset.ontology.getAxioms())

# Get training axioms
ore_dataset = PathDataset(ore_owl)
train_dataset = PathDataset(train_owl)
ore_axioms = set(ore_dataset.ontology.getAxioms())
train_axioms = set(train_dataset.ontology.getAxioms())

# Remove validation and testing axioms from training axioms
full_train_axioms = ore_axioms.union(train_axioms)
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

