import torch
import random
import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
#from scipy.sparse.linalg.eigen.arpack import eigsh
import networkx as nx
import torch_geometric
import torch_sparse

def generate_data(number_of_nodes, class_num, link_inclass_prob, link_outclass_prob): # 模拟用的
    adj=torch.zeros(number_of_nodes,number_of_nodes) #n*n adj matrix

    labels=torch.randint(0,class_num,(number_of_nodes,)) #assign random label with equal probability
    labels=labels.to(dtype=torch.long)
    #label_node, speed up the generation of edges
    label_node_dict=dict()

    for j in range(class_num):
            label_node_dict[j]=[]

    for i in range(len(labels)):
        label_node_dict[int(labels[i])]+=[int(i)]
    # label_id -> [client_id ... ....]

    #generate graph
    for node_id in range(number_of_nodes):
                j=labels[node_id]
                for l in label_node_dict:
                    if l==j: # 属于一个class
                        for z in label_node_dict[l]:  #z>node_id,  symmetrix matrix, no repeat
                            if z>node_id and random.random()<link_inclass_prob:
                                adj[node_id,z]= 1
                                adj[z,node_id]= 1
                    else:    # 属于不同class
                        for z in label_node_dict[l]:
                            if z>node_id and random.random()<link_outclass_prob:
                                adj[node_id,z]= 1
                                adj[z,node_id]= 1
                              
    adj=torch_geometric.utils.dense_to_sparse(torch.tensor(adj))[0]

    #generate feature use eye matrix
    features=torch.eye(number_of_nodes,number_of_nodes)

    #seprate train,val,test
    idx_train = torch.LongTensor(range(number_of_nodes//5))
    idx_val = torch.LongTensor(range(number_of_nodes//5, number_of_nodes//2))
    idx_test = torch.LongTensor(range(number_of_nodes//2, number_of_nodes))

    return features.float(), adj, labels, idx_train, idx_val, idx_test

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        # print(type(x)) # 稀疏矩阵
        # print(x.shape) # (140, 1433)
        # print(x) # 文章id -> 字典对应的单词是否存在
        
        # print(type(y)) # <class 'numpy.ndarray'>
        # print(y.shape) # (140, 7)
        # print(y) # 文章id -> 文章对应的类型:生物、化学、环境、计算机
        
        # print(type(allx)) # 稀疏矩阵
        # print(allx.shape) # (1708, 1433)
        # print(allx) # 文章id -> 字典对应的单词是否存在
        
        # print(type(ally)) # 稀疏矩阵
        # print(ally.shape) # (1708, 7)
        # print(ally) # 文章id -> 文章对应的类型:生物、化学、环境、计算机
        
        # print(type(tx)) # 稀疏矩阵
        # print(tx.shape) # (1000, 1433)
        # print(tx) # 文章id -> 字典对应的单词是否存在
        
        # print(type(ty)) # 稀疏矩阵
        # print(ty.shape) # (1000, 7)
        # print(ty) # 文章id -> 文章对应的类型:生物、化学、环境、计算机
        
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        # print(f"test_idx_reorder:{test_idx_reorder}") # 应该是那些需要预测的文章id
        test_idx_range = np.sort(test_idx_reorder)
        # print(f"test_idx_range:{test_idx_range}") # [1708，2707]，预测1000篇文章的id
        
        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil() # # torch.Size([2708, 1433]) 按行堆叠，其他维度需要相同. allx（其中有带标签的，也有不带标签的） + tx = X
        features[test_idx_reorder, :] = features[test_idx_range, :] # 对测试集进行reorder，规则是将[1708,2707]的文章id词典 转移到test_idx_reorder[id-1708]对应的索引行上去
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        
        # print(type(adj)) # 稀疏矩阵 
        # print(adj.shape) # (2708, 2708)
        # print(adj) # 文章id 之间的引用关系

        labels = np.vstack((ally, ty)) # Y
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        number_of_nodes=adj.shape[0]


        idx_test = test_idx_range.tolist() # [1708，2707]
        idx_train = range(len(y)) # (140, 7)
        idx_val = range(len(y), len(y)+500) # [140,640]

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        
        # print(idx_test.shape) # [1708，2707]
        # print(idx_train.shape) # (140, 7)
        # print(idx_val.shape) # [140,640]

        #features = normalize(features) #cannot converge if use SGD, why??????????
        #adj = normalize(adj)    # no normalize adj here, normalize it in the training process


        features=torch.tensor(features.toarray()).float()
        adj = torch.tensor(adj.toarray())
        # print(adj.shape) # torch.Size([2708, 2708])
        adj = torch_sparse.tensor.SparseTensor.from_edge_index(torch_geometric.utils.dense_to_sparse(adj)[0])
        #edge_index=torch_geometric.utils.dense_to_sparse(torch.tensor(adj.toarray()))[0]
        # print(adj.to_torch_sparse_coo_tensor())
    #     tensor(indices=tensor([[   0,    0,    0,  ..., 2707, 2707, 2707],
    #                    [ 633, 1862, 2582,  ...,  598, 1473, 2706]]),
    #    values=tensor([1., 1., 1.,  ..., 1., 1., 1.]),
    #    size=(2708, 2708), nnz=10556, layout=torch.sparse_coo)

        labels=torch.tensor(labels) # allY + ty = Y
        labels=torch.argmax(labels,dim=1) # 每行最大值出现的列是多少，即每篇文章最可能的类型是什么
    elif dataset_str in ['ogbn-arxiv', 'ogbn-products', 'ogbn-mag', 'ogbn-papers100M']: #'ogbn-mag' is heteregeneous
        #from ogb.nodeproppred import NodePropPredDataset
        from ogb.nodeproppred import PygNodePropPredDataset

        # Download and process data at './dataset/.'

        #dataset = NodePropPredDataset(name = dataset_str, root = 'dataset/')
        dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=torch_geometric.transforms.ToSparseTensor())

        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        data = dataset[0]
        
        features = data.x #torch.tensor(graph[0]['node_feat'])
        labels = data.y.reshape(-1) #torch.tensor(graph[1].reshape(-1))
        adj = data.adj_t.to_symmetric()
        #edge_index = torch.tensor(graph[0]['edge_index'])
        #adj = torch_geometric.utils.to_dense_adj(torch.tensor(graph[0]['edge_index']))[0]

    return features.float(), adj, labels, idx_train, idx_val, idx_test

np.random.seed(42)
torch.manual_seed(42)
#'cora', 'citeseer', 'pubmed' #simulate #other dataset twitter, 
dataset_name="cora"#'ogbn-arxiv'

if dataset_name == 'simulate':
    number_of_nodes=200
    class_num=3
    link_inclass_prob=10/number_of_nodes  #when calculation , remove the link in itself
    #EGCN good when network is dense 20/number_of_nodes  #fails when network is sparse. 20/number_of_nodes/5

    link_outclass_prob=link_inclass_prob/20


    features, adj, labels, idx_train, idx_val, idx_test = generate_data(number_of_nodes,  class_num, link_inclass_prob, link_outclass_prob)               
else:
    features, adj, labels, idx_train, idx_val, idx_test = load_data(dataset_name)
    class_num = labels.max().item() + 1

# print(type(features))
# print(features.shape) # torch.Size([2708, 1433])
# indices = torch.nonzero(features)
# row = indices[:,0]
# col = indices[:,1]

# print(f"row:{row}") 
# print(f"col:{col}") 
# print(features) # 每篇文章出现了哪些单词，单词数来自于一本词典，这本词典中总共有1433个单词 


# 总共有2708个点，代表论文，论文中出现的单词在某本词典中，feature就是每篇论文中出现了哪些在词典中的单词。label就是这篇论文的类型。论文
# 之间可能会产生引用叫做邻接边。