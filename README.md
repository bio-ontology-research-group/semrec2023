# semrec2023

This repository contains the code and models/embeddings for the Semantic Reasoning Evaluataion Challenge 2023.


## Creating a virtual environment

You can create an Anaconda virtual environment as follows:

```
conda env create -f environment.yml
conda activate semrec
```

We use mOWL version 0.2.0 (not yet as PyPi package), which can be downloaded from 
this [link](https://bio2vec.cbrc.kaust.edu.sa/data/mowl/mowl-borg-0.2.0.tar.gz). To install it:

```
pip install mowl-borg-0.2.0.tar.gz
```

## Embeddings/Models

To access to the trained embeddings/models click in this [link](https://bio2vec.cbrc.kaust.edu.sa/data/mowl/semrec2023.tar.gz)

After decompressing the file, copy all the contents under `use_cases/` into the folder `use_cases/` of this repository.

# Adapting the data

## Adapting data formats for ORE datasets

Validation and testing ontologies have been provided in Functional Syntax. We have changed the format to OWL syntax using the script `src/wrap_ore_ont.py`

```
python wrap_ore_ont.py <input file> [train|valid|test]
```


## Cleaning the training ORE datasets

For each dataset, we made sure to remove overlapping information between "training" and "validation and testing" ontologies using the script `src/clean_ore_dataset.py`.

```
python clean_ore_dataset.py <ore dataset number>
```


## Adapting data formats for OWL2Bench datasets

Validation and testing ontologies have been provided in Functional Syntax. We have changed the format to OWL syntax using the script `src/wrap_owl2bench_ont.py`

```
python wrap_owl2bench_ont.py <input file> [train|valid|test]
```


## Cleaning the training OWL2Bench datasets

For each dataset, we made sure to remove overlapping information between "training" and "validation and testing" ontologies using the script `src/clean_owl2bench_dataset.py`.

```
python clean_owl2bench_dataset.py <owl2bench dataset number>
```



## Adapting data formats for CaliGraph 

Training, validation and testing ontologies have been provided as NTriples files. We make sure the NTriples file are correctly formated (one line per triple) with the script `src/nt_format.py`

```
python nt_format.py <input file>.nt
```

Training, validation and testing ontologies have been provided as NTriples files. We have changed the format to OWL syntax using the script `src/nt_to_owl.py`

```
python nt_to_owl.py <input file>_corrected.nt
```


## Cleaning the training Caligraph datasets

For each dataset, we made sure to remove overlapping information between "training" and "validation and testing" ontologies using the script `src/clean_caligraph_dataset.py`.

```
python clean_caligraph_dataset.py <caligraph dataset number>
```
