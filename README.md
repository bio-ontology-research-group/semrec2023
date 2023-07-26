# semrec2023


## Adapting data formats

Validation and testing ontologies have been provided in Functional Syntax. We have changed the format to OWL syntax using the script `src/wrap_ont.py`


## Cleaning the training dataset

For each dataset, we made sure to remove overlapping information between "training" and "validation and testing" ontologies using the script `src/clean_[dataset_name]_dataset.py`.
