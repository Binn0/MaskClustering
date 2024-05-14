import networkx as nx
from graph.node import Node
import torch
from IPython import embed

def cluster_into_new_nodes(iteration, old_nodes, graph):
    new_nodes = [] # len: 471
    for component in nx.connected_components(graph):
        node_info = (iteration, len(new_nodes))
        new_nodes.append(Node.create_node_from_list([old_nodes[node] for node in component], node_info))

    # print("cluster_into_new_nodes: 8 embed")
    # embed()
    return new_nodes


def update_graph(nodes, observer_num_threshold, connect_threshold):
    '''
        update view consensus rates between nodes and return a new graph
    '''
    # connect_threshold: 0.9
    node_visible_frames = torch.stack([node.visible_frame for node in nodes], dim=0) # torch.Size([632, 83]),M[i,j] stores whether node i and node j both appear, 0 or 1
    node_contained_masks = torch.stack([node.contained_mask for node in nodes], dim=0) # torch.Size([632, 661]), 0/1

    
    '''
    # torch.Size([633, 633])
    observer_nums[0]
    tensor([12.,  1.,  3.,  7.,  1., 10.,  1.,  5.,  6.,  1.,  5.,  7., 10.,  1.,
         1.,  5.,  9.,  1.,  1.,  7.,  6.,  1., 12.,  5.,  1.,  1.,  8.,  4.,
         7., 10., 12.,  1.,  6.,  0.,  8.,  0., 12.,  7.,  1.,  6.,  2.,  0.,
         8., 10.,  8., 11.,  3.,  7.,  2., 11., 10.,  8.,  1.,  0.,  6., 11.,
         8.,  3.,  0.,  2., 11.,  0., 10.,  6.,  8., 11.,  2.,  8.,  3., 11.,
         2.,  1., 10., 11.,  2.,  8.,  6.,  8.,  3.,  7., 11.,  8.,  3.,  8.,
        10.,  8.,  7., 11.,  7.,  8., 11.,  8., 10.,  8.,  7., 11.,  2.,  5.,
        11.,  6.,  8.,  1.,  8.,  9.,  7.,  4.,  9.,  7., 11.,  1.,  7.,  8.,
         8., 11.,  8.,  7.,  4.,  5., 11.,  8.,  2.,  8.,  7.,  9., 11.,  5.,
         8.,  8.,  4.,  4., 11.,  7.,  7.,  5.,  7., 11.,  1.,  9.,  4.,  8.,
         3.,  4.,  4.,  7.,  6., 11.,  8.,  5.,  6.,  9.,  4.,  9., 11.,  4.,
         4.,  2.,  7.,  5.,  5., 11.,  3.,  8.,  5.,  4., 11.,  4.,  1.,  4.,
         8.,  4., 11.,  5.,  2., 11.,  0.,  4.,  4.,  2.,  5.,  7.,  1.,  0.,
         4.,  0.,  3.,  5.,  5.,  1.,  0.,  0.,  4.,  4.,  1.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         6.,  0.,  4.,  0.,  5.,  7.,  1.,  5.,  0., 11.,  7.,  1.,  1.,  6.,
         0.,  8.,  1.,  5.,  6.,  8.,  1.,  0., 11.,  6.,  3.,  5.,  1.,  7.,
         0.,  3.,  8.,  5., 12.,  0., 11.,  8.,  0.,  2.,  8.,  0.,  3.,  6.,
         0., 11.,  8., 10.,  7.,  1., 11.,  8.,  3.,  3., 11.,  8.,  1.,  8.,
        11.,  7.,  3.,  8., 11.,  1., 11.,  2.,  2.,  8.,  6.,  8., 11.,  2.,
         4.,  2.,  5.,  8.,  9., 11.,  2.,  4.,  5.,  8.,  2., 10.,  8.,  2.,
         5.,  3.,  8.,  2., 10.,  8.,  0.,  6.,  8.,  2., 11.,  8., 11.,  3.,
         7., 11.,  8.], device='cuda:0')
    '''
    observer_nums = torch.matmul(node_visible_frames, node_visible_frames.transpose(0,1)) # M[i,j] stores the number of frames that node i and node j both appear
    
    # torch.Size([633, 633]), 这里的633数量会变化，根据nodes数量，每次迭代会变
    supporter_nums = torch.matmul(node_contained_masks, node_contained_masks.transpose(0,1)) # M[i,j] stores the number of frames that supports the merging of node i and node j

    # torch.Size([633, 633])
    view_concensus_rate = supporter_nums / (observer_nums + 1e-7) # floats数据，1.0/0.8333/0.500

    disconnect = torch.eye(len(nodes), dtype=bool).cuda()
    
    # torch.Size([633, 633]) 存在的True和Flase
    disconnect = disconnect | (observer_nums < observer_num_threshold) # node pairs with less than observer_num_threshold observers are disconnected

    A = view_concensus_rate >= connect_threshold
    A = A & ~disconnect
    A = A.cpu().numpy() # shape: (633, 633)

    # G.nodes: [0, 1, 2, ..., 632]
    # G.edges: [(3, 11), (5, 22), (5, 30), (5, 36),...,]
    G = nx.from_numpy_array(A) # Graph with 633 nodes and 1980 edges

    # print("update_graph: 7 embed")
    # embed()
    return G


def iterative_clustering(nodes, observer_num_thresholds, connect_threshold, debug):
    if debug:
        print('====> Start iterative clustering')
    for iterate_id, observer_num_threshold in enumerate(observer_num_thresholds):
        if debug:
            print(f'Iterate {iterate_id}: observer_num', observer_num_threshold, ', number of nodes', len(nodes))
        graph = update_graph(nodes, observer_num_threshold, connect_threshold)
        nodes = cluster_into_new_nodes(iterate_id+1, nodes, graph)
    return nodes