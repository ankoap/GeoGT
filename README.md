# GeoGT: Geometry-Enhanced Graph Transformer for Molecular Ground-State Conformation Prediction

The implementation of GeoGT. 

## Environment

### Requirements

* python == 3.10
* pytorch == 2.1.0
* cuda == 12.1

### Install via Conda

```bash
conda env create -f environment.yaml
conda activate geogt
```

## Datasets

### official datasets 

The official datasets can be found at [Molecule3D](https://github.com/divelab/MoleculeX/tree/molx/Molecule3D) and [Qm9](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904).


## Experiments

### Molecule Tokenization

#### Mole-BERT Tokenizer Training

```bash
cd /path/to/GeoGT
bash experiments/tokenizer_training/mol_bert_tokenizer_training.sh
```

#### Tokenize Molecules

```bash
cd /path/to/GeoGT   
python -m tokenize_mole \
    --save_dir /path/to/save/tokenized/dataset/ \
    --dataset_name Molecule3D (or Qm9) \
    --mode random (or scaffold) \
    --tokenizer_checkpoint /path/to/tokenizer/checkpoint \
```

### Molecular Ground-State Conformation Prediction

#### Training

```bash
cd /path/to/GeoGT  
bash experiments/conformer_prediction/geogt_for_conformer_prediction.sh
```

#### Evaluation

```bash
cd /path/to/GeoGT   
python -m evaluate \
    --data_dir /path/to/data_dir/ \
    --dataset Qm9 \
    --split test \
    --log_file /path/to/log_file.txt \
    --method GeoGT \
    --GeoGT_checkpoint /path/to/GeoGT_checkpoint \
    --MoleBERT_Tokenizer_checkpoint  /path/to/MoleBERT_Tokenizer_checkpoint\
    --device cuda:0 \ 
    --batch_size 100 \
```

