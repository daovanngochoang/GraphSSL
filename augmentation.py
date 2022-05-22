import random
import numpy as np
import torch
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
import matplotlib.pyplot as plt
def edge_deletion(data):#torch_geometric.data.Data type
    '''
    input a graph, randomly select an edge and remove it from the graph, return the processed graph
    torch tensor:
    :param data.x: [num_nodes,num_node_features]
    :param data.edge_index: [2,num_edges]
    :param data.edge_attr: [num_edges, num_edge_features] for now we do not consider edge features
    :param data.y: [graph_label_dimension]
    :return: torch_geometric.data.Data
    '''

    num_edges = data.num_edges
    if num_edges==0:
        return data
    random_number = random.randint(0, num_edges-1)

    new_edge_index = data.edge_index.t().detach().cpu().numpy().tolist()

    start = new_edge_index[random_number][0]
    end = new_edge_index[random_number][1]
    new_edge_index.remove([start,end])
    new_edge_index.remove([end,start])


    new_edge_index = torch.Tensor(new_edge_index).t().contiguous().long()

    new_graph = Data(x=data.x,edge_index=new_edge_index,y=data.y, edge_attr=None)

    return new_graph

def node_deletion(data):#torch_geometric.data.Data type
    '''
    input a graph, randomly  select  an  node  and  remove  it from the graph; remove all edges connecting to this node, return the processed graph
    torch tensor:
    :param data.x: [num_nodes,num_node_features]
    :param data.edge_index: [2,num_edges]
    :param data.edge_attr: [num_edges, num_edge_features] for now we do not consider edge features
    :param data.y: [graph_label_dimension]
    :return: torch_geometric.data.Data
    '''

    num_nodes = data.num_nodes
    if num_nodes==0:
        return data
    if data.num_edges==0:
        return data

    random_number = random.randint(0, num_nodes-1)


    node_features1 = data.x[:random_number, :]
    node_features2 = data.x[random_number+1:, :]
    new_node_features = torch.cat((node_features1,node_features2), dim=0)

    #undirected graph means two directed edges
    loc = (data.edge_index==random_number).nonzero()
    start = random_number
    rol = 1 - loc[:,0]
    col = loc[:,1]

    end = data.edge_index[rol,col].long()
    edge_index_list = data.edge_index.t().long().detach().cpu().numpy().tolist()
    for endi in end:
        if [start,endi.item()] in edge_index_list:
            edge_index_list.remove([start,endi.item()])

        if [endi.item(),start] in edge_index_list:
            edge_index_list.remove([endi.item(),start])


    new_edge_index = torch.Tensor(edge_index_list).t().contiguous().long()
    new_edge_index  =torch.where(new_edge_index>random_number, new_edge_index-1, new_edge_index) #an intermediate node is deleted, the index of nodes after it should -1
    new_graph = Data(x=new_node_features,y=data.y,edge_index=new_edge_index, edge_attr=None)
    return new_graph

def edge_insertion(data):#torch_geometric.data.Data type
    '''
    input a graph, randomly  select two nodes,  if they are not directly connected but there is a path between them,add an edge between these two nodes, return the processed graph
    torch tensor:
    :param data.x: [num_nodes,num_node_features]
    :param data.edge_index: [2,num_edges]
    :param data.edge_attr: [num_edges, num_edge_features] for now we do not consider edge features
    :param data.y: [graph_label_dimension]
    :return: torch_geometric.data.Data
    '''


    num_nodes = data.x.shape[0]
    G = to_networkx(data)
    succeed=False

    random_nodes = random.sample(range(0,num_nodes),2)

    edge_index_list = data.edge_index.t().detach().cpu().numpy().tolist()

    if nx.has_path(G,source=random_nodes[0],target=random_nodes[1]) and nx.has_path(G,source=random_nodes[1],target=random_nodes[0]):

        edge_between_nodes = [random_nodes[0],random_nodes[1]] in edge_index_list
        if not edge_between_nodes:

            edge1 = torch.Tensor([random_nodes[0],random_nodes[1]]).unsqueeze(dim=1).long()
            edge2 = torch.Tensor([random_nodes[1],random_nodes[0]]).unsqueeze(dim=1).long()
            new_edge_index = torch.cat((data.edge_index,edge1,edge2),dim=1)

            succeed=True

    if succeed==False:
        return edge_deletion(data)
    else:
        data.edge_index = new_edge_index.contiguous()
        data.edge_attr=None

        return data

def is_onehot(x):
    flag=False
    num_one = 0
    num_zero=0
    for i in x:
        if i==1:
            num_one+=1
        elif i==0:
            num_zero+=1
        else:
            flag=False
    if num_one==1 and num_zero==x.shape[0]-1:
        flag=True
    return flag

def get_discrete_attr_for_inserted_node(component_features):
    '''
    find mode and transform to onehot
    :param component_features : [num_node_in_component, num_features] onehot
    :return: [1,num_features] onehot
    '''
    com = torch.argmax(component_features,dim=1)
    list = com.detach().cpu().numpy().tolist()
    list_set=set(list)
    frequency_dict={}
    for i in list_set:
        frequency_dict[i]=list.count(i)
    grade_mode=[]
    for key,value in frequency_dict.items():
        if value==max(frequency_dict.values()):
            grade_mode.append(key)
    random_number = random.randint(0,len(grade_mode)-1)
    mode = grade_mode[random_number]
    one_hot = torch.zeros(1, component_features.shape[1]).scatter_(1, torch.LongTensor([[mode]]), 1)
    return one_hot

def node_insertion(data):#torch_geometric.data.Data type
    '''
    input a graph, randomly  select  a  strongly-connected sub-graph S, remove all edges in S, add a node n, and add an edge between n and each node in S, return the processed graph
    torch tensor:
    :param data.x: [num_nodes,num_node_features]
    :param data.edge_index: [2,num_edges]
    :param data.edge_attr: [num_edges, num_edge_features] for now we do not consider edge features
    :param data.y: [graph_label_dimension]
    :return: torch_geometric.data.Data
    '''

    num_nodes = data.x.shape[0]
    G = to_networkx(data)

    con = nx.strongly_connected_components(G)

    component_list = []
    for component in list(con):
        if len(component)>1:
            component_list.append(list(component))
    if len(component_list)<1:
        return data
    random_number = random.randint(0,len(component_list)-1)

    G.add_node(num_nodes)

    discrete = is_onehot(data.x[0])
    if discrete==False:
        new_node = torch.mean(data.x[component_list[random_number],:], dim=0)
        new_node = new_node.unsqueeze(dim=0)
    else:
        component_features = data.x[component_list[random_number],:]
        new_node = get_discrete_attr_for_inserted_node(component_features)

    new_node_features = torch.cat((data.x, new_node),dim=0)

    edge_index_list = data.edge_index.t().long().detach().cpu().numpy().tolist()
    all_edges_in_component = []
    for u in component_list[random_number]:
        for v in component_list[random_number]:
            if [u,v] in edge_index_list:
                all_edges_in_component.append((u,v))


    G.remove_edges_from(all_edges_in_component)

    for node in component_list[random_number]:
        G.add_edge(node,num_nodes)
        G.add_edge(num_nodes,node)

    new_edge_index = torch.Tensor(list(G.edges)).t().contiguous().long()

    new_graph = Data(x=new_node_features, y = data.y, edge_index= new_edge_index, edge_attr=None)
    return new_graph


def random_augmentation(batch_graph):
    '''

    :param batch_graph: a list(batch) of graphs
    :return: a list(batch) of graphs
    '''
    new_graph_list = []

    for graph in batch_graph:
        random_number = random.randint(0,3)
        new_graph = graph

        if random_number==0:
            new_graph = edge_deletion(new_graph)
        elif random_number==1:
            new_graph = node_deletion(new_graph)
        elif random_number==2:
            new_graph = edge_insertion(new_graph)
        elif random_number==3:
            new_graph = node_insertion(new_graph)

        new_graph_list.append(new_graph)

    return new_graph_list


