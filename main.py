import torch
from utils.config import get_dataset, get_args
from utils.post_process import post_process
from graph.construction import mask_graph_construction
from graph.iterative_clustering import iterative_clustering
from tqdm import tqdm
import os
from IPython import embed

def main(args):
    # scene0549_00为例子
    dataset = get_dataset(args)
    scene_points = dataset.get_scene_points() # (161973, 3)
    frame_list = dataset.get_frame_list(args.step) # 一个一个的frames
    print("agrs.step: ",args.step) # step: 10
    print("frame_list: ", frame_list) # [0, 10, 20, ..., 820] 总共83个frame_ID
    # if os.path.exists(os.path.join(dataset.object_dict_dir, args.config, f'object_dict.npy')):
    #     return

    with torch.no_grad():
        nodes, observer_num_thresholds, mask_point_clouds, point_frame_matrix = mask_graph_construction(args, scene_points, frame_list, dataset)

        # print("mask_graph_construction: 6 embed")
        # embed()

        # len: 68; [<graph.node.Node object at 0x7f8b945b7990>,...,]每一个是个Node对象
        object_list = iterative_clustering(nodes, observer_num_thresholds, args.view_consensus_threshold, args.debug)
        '''
        Iterate 0: observer_num 24.0 , number of nodes 632
        Iterate 1: observer_num 19.0 , number of nodes 473
        Iterate 2: observer_num 16.0 , number of nodes 437
        Iterate 3: observer_num 13.0 , number of nodes 359
        Iterate 4: observer_num 12.0 , number of nodes 307
        Iterate 5: observer_num 10.0 , number of nodes 282
        Iterate 6: observer_num 9.0 , number of nodes 238
        Iterate 7: observer_num 8.0 , number of nodes 224
        Iterate 8: observer_num 7.0 , number of nodes 184
        Iterate 9: observer_num 6.0 , number of nodes 162
        Iterate 10: observer_num 5.0 , number of nodes 134
        Iterate 11: observer_num 4.0 , number of nodes 104
        Iterate 12: observer_num 4.0 , number of nodes 85
        Iterate 13: observer_num 3.0 , number of nodes 85
        Iterate 14: observer_num 3.0 , number of nodes 76
        Iterate 15: observer_num 2.0 , number of nodes 76
        Iterate 16: observer_num 2.0 , number of nodes 68
        '''
        # print("iterative_clustering: 9 embed")
        # embed()

        post_process(dataset, object_list, mask_point_clouds, scene_points, point_frame_matrix, frame_list, args)

if __name__ == '__main__':
    args = get_args()
    seq_name_list = args.seq_name_list.split('+')

    for seq_name in tqdm(seq_name_list):
        args.seq_name = seq_name
        main(args)