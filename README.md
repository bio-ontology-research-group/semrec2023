# semrec2023


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
