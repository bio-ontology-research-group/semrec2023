import mowl
mowl.init_jvm("10g")
from mowl.datasets import PathDataset
from mowl.owlapi import OWLAPIAdapter

from org.semanticweb.owlapi.formats import OWLXMLDocumentFormat
from org.semanticweb.owlapi.model import IRI
from java.util import HashSet

import os
import sys


caligraph_number = int(sys.argv[1])
root = f"../use_cases/clg_10e{caligraph_number}/data"
if caligraph_number < 4 or caligraph_number > 5:
    raise ValueError("caligraph_number must be 4 or 5")

caligraph_owl = f"{root}/clg_10e{caligraph_number}.owl"
train_owl = f"{root}/clg_10e{caligraph_number}-train_corrected.owl"
valid_owl = f"{root}/clg_10e{caligraph_number}-val_corrected.owl"
test_owl = f"{root}/clg_10e{caligraph_number}-test_corrected.owl"

output_file = f"{root}/clg_10e{caligraph_number}_cleaned.owl"

# Get validation and testing axioms
valid_dataset = PathDataset(valid_owl)
test_dataset = PathDataset(test_owl)
valid_axioms = set(valid_dataset.ontology.getAxioms())
test_axioms = set(test_dataset.ontology.getAxioms())

# Get training axioms
caligraph_dataset = PathDataset(caligraph_owl)
train_dataset = PathDataset(train_owl)
caligraph_axioms = set(caligraph_dataset.ontology.getAxioms())
train_axioms = set(train_dataset.ontology.getAxioms())

# Remove validation and testing axioms from training axioms
full_train_axioms = caligraph_axioms.union(train_axioms)
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

