
# Non-Backtracking Graph Neural Network for Node Classification

This repository is the official implementation of 'Non-Backtracking Graph Neural Network' for node classification task. 


## Requirements

This codebase is written for python3 (used python 3.10.10 while implementing).
To install requirements, run:

```setup
conda env create --file environment.yaml
```

## Training

To train the best models for each dataset, run this command:


**Dataset: Cora, Model: GCN+NBA**
```
python train.py --num_steps 1000 --dataset cora --model GCN --nbtm --name nbtm_GCN_l3 --num_layers 3 
```

**Dataset: CiteSeer, Model: GraphSAGE+NBA+LapPE**
```
python train.py --num_steps 1000 --dataset citeseer --model SAGE --nbtm --PE --name PE_nbtm_SAGE_l3 --num_layers 3 
```

**Dataset: PubMed, Model: GraphSAGE+NBA**
```
python train.py --num_steps 1000 --dataset pubmed --model SAGE --nbtm --name nbtm_SAGE_l3 --num_layers 3
```

**Dataset: Texas, Model: GraphSAGE+NBA+LapPE**
```
python train.py --num_steps 100 --dataset texas --model SAGE --nbtm --PE --name PE_nbtm_SAGE_l3 --num_layers 3
```

**Dataset: Wisconsin, Model: GraphSAGE+NBA**
```
python train.py --num_steps 100 --dataset wisconsin --model SAGE --nbtm True --name nbtm_SAGE_l4 --num_layers 4
```

**Dataset: Cornell, Model: GraphSAGE+NBA**
```
python train.py --num_steps 100 --dataset cornell --model SAGE --nbtm True --name nbtm_SAGE_l4 --num_layers 4 
```

## Reading Results

If you want to see the existing results of trained models, run:
```
python read_results.py
```

## Results
In our experiments, we employed various GNN architectures (Kipf & Welling,2016; Hamilton et al., 2017; Velickovic et al., 2018; Li et al., 2016), along with their corresponding NBA versions,

| **Model**      | **Cora**      | **CiteSeer**  | **$PubMed**   | **Texas**     | **Wisconsin** | **Cornell**   |
|----------------|---------------|---------------|---------------|---------------|---------------|---------------|
| **GCN**        | 0.8658±0.0060 | 0.7532±0.0134 | 0.8825±0.0042 | 0.6162±0.0634 | 0.6059±0.0438 | 0.5946±0.0662 |
| **+NBA**       | 0.8722±0.0095 | 0.7585±0.0175 | 0.8826±0.0044 | 0.7108±0.0796 | 0.7471±0.0386 | 0.6108±0.0614 |
| **+NBA+LapPE** | 0.8720±0.0129 | 0.7609±0.0186 | 0.8827±0.0048 | 0.6811±0.0595 | 0.7471±0.0466 | 0.6378±0.0317 |
| **GraphSAGE**  | 0.8632±0.0158 | 0.7559±0.0161 | 0.8864±0.0030 | 0.7108±0.0556 | 0.7706±0.0403 | 0.6027±0.0625 |
| **+NBA**       | 0.8702±0.0083 | 0.7586±0.0213 | 0.8871±0.0044 | 0.7270±0.0905 | 0.7765±0.0508 | 0.6459±0.0691 |
| **+NBA+LapPE** | 0.8650±0.0120 | 0.7621±0.0172 | 0.8870±0.0037 | 0.7486±0.0612 | 0.7647±0.0531 | 0.6378±0.0544 |
| **GAT**        | 0.8694±0.0119 | 0.7463±0.0159 | 0.8787±0.0046 | 0.6054±0.0386 | 0.6000±0.0491 | 0.4757±0.0614 |
| **+NBA**       | 0.8722±0.0120 | 0.7549±0.0171 | 0.8829±0.0043 | 0.6622±0.0514 | 0.7059±0.0562 | 0.5838±0.0558 |
| **+NBA+LapPE** | 0.8692±0.0098 | 0.7561±0.0175 | 0.8822±0.0047 | 0.6730±0.0348 | 0.7314±0.0531 | 0.5784±0.0640 |
| **GatedGCN**   | 0.8477±0.0156 | 0.7325±0.0192 | 0.8671±0.0060 | 0.6108±0.0652 | 0.5824±0.0641 | 0.5216±0.0987 |
| **+NBA**       | 0.8523±0.0095 | 0.7405±0.0187 | 0.8661±0.0035 | 0.6162±0.0490 | 0.6431±0.0356 | 0.5649±0.0532 |
| **+NBA+LapPE** | 0.8517±0.0130 | 0.7379±0.0193 | 0.8661±0.0047 | 0.6243±0.0467 | 0.6569±0.0310 | 0.5405±0.0785 |


## Contributing

This project is licensed under the terms of the MIT license.
