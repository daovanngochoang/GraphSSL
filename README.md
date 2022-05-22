# Contrastive Self-supervised Learning for Graph Classification
This is a PyTorch implementation of the methods proposed in the paper.


## Requirements
* python3.7
* pytorch==1.4.0
* CUDA==10.1
* torch-scatter==2.0.4
* torch-sparse==0.6.1
* torch-cluster==1.5.4
* torch-geometric==1.4.2

### Datasets
Graph classification benchmarks are publicly available at [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).

This folder contains the following comma separated text files (replace DS by the name of the dataset):

**n = total number of nodes**

**m = total number of edges**

**N = number of graphs**

**(1) DS_A.txt (m lines)** 

*sparse (block diagonal) adjacency matrix for all graphs, each line corresponds to (row, col) resp. (node_id, node_id)*

**(2) DS_graph_indicator.txt (n lines)**

*column vector of graph identifiers for all nodes of all graphs, the value in the i-th line is the graph_id of the node with node_id i*

**(3) DS_graph_labels.txt (N lines)** 

*class labels for all graphs in the dataset, the value in the i-th line is the class label of the graph with graph_id i*

**(4) DS_node_labels.txt (n lines)**

*column vector of node labels, the value in the i-th line corresponds to the node with node_id i*

There are OPTIONAL files if the respective information is available:

**(5) DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)**

*labels for the edges in DS_A_sparse.txt* 

**(6) DS_edge_attributes.txt (m lines; same size as DS_A.txt)**

*attributes for the edges in DS_A.txt* 

**(7) DS_node_attributes.txt (n lines)** 

*matrix of node attributes, the comma seperated values in the i-th line is the attribute vector of the node with node_id i*

**(8) DS_graph_attributes.txt (N lines)** 

*regression values for all graphs in the dataset, the value in the i-th line is the attribute of the graph with graph_id i*


### Run
#### CSSL-Pretrain

In this approach, we first pretrain the graph encoder with CSSL and then finetune it in a supervised way.

For pretraining, we can execute the following command:

```python
python main_moco.py --gpu 0 --cos --dataset NCI1 --lr 1e-5 --epochs 1000 -b 16
```

After pretraining, the model will be saved in the folder './results/'+args.dataset+'/'+str(args.batch_size)'. We can then load the model and finetune it. For example, we can execute the following command:

```python
python finetune.py --dataset NCI1 --device cuda:0 --resume ./results/NCI1/16/checkpoint_00001.pth.tar --batch_size 16
```

#### CSSL-Freeze

The pretraining process is shown above. After pre-training, we fix the graph encoder and train an MLP to conduct classification. For example, we can execute the following command:

```python
python cls.py --dataset NCI1 --device cuda:0 --resume ./results/NCI1/16/checkpoint_00001.pth.tar --batch_size 16
```

#### CSSL-Reg

In this method, we view CSSL task as a regularizer. 

If we train and test on a single dataset, execute:

```python
python reg.py --dataset NCI1 --gpu 0 -b 16
```

If we train on all datasets and test on a specific dataset, execute:

```python
python reg_all.py --dataset all --test_dataset NCI1 --gpu 0 -b 16
```


