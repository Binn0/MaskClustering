import torch
import numpy as np
from tqdm import tqdm
from utils.mask_backprojection import frame_backprojection
from graph.node import Node
from IPython import embed
def mask_graph_construction(args, scene_points, frame_list, dataset):
    '''
        Construct the mask graph:
        1. Build the point in mask matrix. (To speed up the following computation of view consensus rate.)
        2. For each mask, compute the frames that it appears and the masks that contains it. Concurrently, we judge whether this mask is undersegmented.
        3. Build the nodes in the graph.
    '''
    if args.debug:
        print('start building point in mask matrix')
    boundary_points, point_in_mask_matrix, mask_point_clouds, point_frame_matrix, global_frame_mask_list = build_point_in_mask_matrix(args, scene_points, frame_list, dataset)
    visible_frames, contained_masks, undersegment_mask_ids = process_masks(frame_list, global_frame_mask_list, point_in_mask_matrix, boundary_points, mask_point_clouds, args)
    observer_num_thresholds = get_observer_num_thresholds(visible_frames)
    nodes = init_nodes(global_frame_mask_list, visible_frames, contained_masks, undersegment_mask_ids, mask_point_clouds)

    return nodes, observer_num_thresholds, mask_point_clouds, point_frame_matrix

def build_point_in_mask_matrix(args, scene_points, frame_list, dataset):
    '''
        To speed up the view consensus rate computation, we build a 'point in mask' matrix by a trade-off of space for time. This matrix is of size (scene_points_num, frame_num). For point i and frame j, if point i is in the k-th mask in frame j, then M[i,j] = k. Otherwise, M[i,j] = 0. (Note that mask id starts from 1).

        Returns:
            boundary_points: a set of points that are contained by multiple masks in a frame and thus are on the boundary of the masks. We will not consider these points in the following computation of view consensus rate.
            point_in_mask_matrix: the 'point in mask' matrix.
            mask_point_clouds: a dict where each key is the mask id in a frame, and the value is the point ids that are in this mask.
            point_frame_matrix: a matrix of size (scene_points_num, frame_num). For point i and frame j, if point i is visible in frame j, then M[i,j] = True. Otherwise, M[i,j] = False.
            global_frame_mask_list: a list of masks in the whole sequence. Each tuple contains the frame id and the mask id in this frame.
    '''
    # scene_points.shape: torch.Size([161973, 3])
    scene_points_num = len(scene_points) # 161973
    frame_num = len(frame_list) # 83

    scene_points = torch.tensor(scene_points).float().cuda() # np -> tensor; torch.Size([161973, 3])
    boundary_points = set() # 也就是mask 的边界的那些点; len: 13317, 存的是point ID， 如131016, 131017, 98247等

    '''
    point_in_mask_matrix.shape: (161973, 83)
    print(point_in_mask_matrix[0])
    [13  0 11  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
        0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
        0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10
        0  0  0  0  0  0  0  0  0  0  0]
    '''
    point_in_mask_matrix = np.zeros((scene_points_num,  frame_num), dtype=np.uint16) # 0 means invisible，存储point在frame当中的哪个mask当中
    
    '''
    point_frame_matrix.shape: (161973, 83)
    print(point_frame_matrix[0])
    [ True False  True False False False False False False False False False
        False False False False False False False False False False False False
        False False False False False False False False False False False False
        False False False False False False False False False False False False
        False False False False False False False False False False False False
        False False False False False False False False False False False  True
        True  True False False False False False False False False False]
    '''
    point_frame_matrix = np.zeros((scene_points_num, frame_num), dtype=bool) # False means invisible， 表示点在frame中是否可见
    
    '''
    len: 661
    print(global_frame_mask_list[:20])
    [(0, 1), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 11), (0, 13), (10, 1), (10, 2), (10, 3), (10, 5), (10, 6), (10, 7), (10, 9), (10, 10), (10, 12), (10, 13), (20, 1)]
    '''
    global_frame_mask_list = [] # 全局frame—mask列表，一个元组记录的是（frame_id, mask_id）

    '''
    len: 661
    dict_keys(['0_1', '0_3', '0_4', '0_5', '0_6', '0_7', '0_8', '0_11', '0_13', '10_1', '10_2', '10_3', '10_5', '10_6', '10_7', '10_9', '10_10', '10_12', '10_13', '20_1', ...}
    dict_values(每个keys的vlaue对应的len: 38 891 1853 123 259 1484 638 1802 5803 15 32 869 1932 335 76 905 1361 1780 ... , 共661项)
    
    print(mask_point_clouds["0_1"])
    {87424, 87425, 87301, 87302, 88585, 88587, 88335, 85651, 87195, 86557, 87970, 88612, 88106, 88369, 88370, 88371, 88372, 85557, 88373, 87103, 88514, 85958, 86984, 88008, 88010, 87629, 86866, 88160, 89056, 89057, 88937, 86254, 88179, 85620, 86777, 86778, 86779, 88318}
    内容对应的是point id
    '''
    mask_point_clouds = {} # 字典，键为frame和mask的组合，值为该mask中包含的点的ID
    
    iterator = tqdm(enumerate(frame_list), total=len(frame_list)) if args.debug else enumerate(frame_list)
    for frame_cnt, frame_id in iterator:
        # mask_dict：key：maskid；value：mask对应在scene当中的point id集合
        # frame_point_cloud_ids: 用来存储所有有效掩码对应的点ID
        mask_dict, frame_point_cloud_ids = frame_backprojection(dataset, scene_points, frame_id)
        if len(frame_point_cloud_ids) == 0:
            continue
        point_frame_matrix[frame_point_cloud_ids, frame_cnt] = True
        appeared_point_ids = set()
        frame_boundary_point_index = set() # 这个也很好理解，其实就是取各个mask的之间的一个交集，也就是边界的点
        for mask_id, mask_point_cloud_ids in mask_dict.items():
            frame_boundary_point_index.update(mask_point_cloud_ids.intersection(appeared_point_ids))
            mask_point_clouds[f'{frame_id}_{mask_id}'] = mask_point_cloud_ids
            point_in_mask_matrix[list(mask_point_cloud_ids), frame_cnt] = mask_id
            appeared_point_ids.update(mask_point_cloud_ids)
            global_frame_mask_list.append((frame_id, mask_id))
        point_in_mask_matrix[list(frame_boundary_point_index), frame_cnt] = 0
        boundary_points.update(frame_boundary_point_index)

    # print("build_point_in_mask_matrix: 1 embed")
    # embed()
    
    return boundary_points, point_in_mask_matrix, mask_point_clouds, point_frame_matrix, global_frame_mask_list

def init_nodes(global_frame_mask_list, mask_project_on_all_frames, contained_masks, undersegment_mask_ids, mask_point_clouds):
    '''
    nodes[0]
    nodes[0].mask_list: [(0, 1)]
    nodes[0].visible_frame: shape torch.Size([83]), 0 means invisible, 1 means visible
    nodes[0].contained_mask: shape torch.Size([661]), 0 means not contained, 1 means contained
    nodes[0].point_ids: len: 38; {87424, 87425, 87301, 87302, 88585, 88587, 88335, 85651, 87195, 86557, 87970, 88612, 88106, 88369, 88370, 88371, 88372, 85557, 88373, 87103, 88514, 85958, 86984, 88008, 88010, 87629, 86866, 88160, 89056, 89057, 88937, 86254, 88179, 85620, 86777, 86778, 86779, 88318}
    nodes[0].node_info: (0, 0) nodes[1].node_info: (0, 1); nodes[2].node_info: (0, 2) ...
    nodes[0].son_node_info: None; nodes[1].son_node_info: None; 
    '''
    nodes = [] # len: 634; 存储的是Node对象
    for global_mask_id, (frame_id, mask_id) in enumerate(global_frame_mask_list):
        if global_mask_id in undersegment_mask_ids:
            continue
        mask_list = [(frame_id, mask_id)]
        frame = mask_project_on_all_frames[global_mask_id]
        frame_mask = contained_masks[global_mask_id]
        point_ids = mask_point_clouds[f'{frame_id}_{mask_id}']
        node_info = (0, len(nodes))
        node = Node(mask_list, frame, frame_mask, point_ids, node_info, None)
        nodes.append(node)

    # print("init_nodes: 5 embed")
    # embed()
    return nodes

def get_observer_num_thresholds(visible_frames):
    '''
        Compute the observer number thresholds for each iteration. Range from 95% to 0%.
    '''
    # visible_frames: len 661; torch.Size([661, 83]); 0 means invisible, 1 means visible
    '''
    observer_num_matrix[0]
    tensor([11.,  1.,  3.,  0.,  6.,  1.,  9.,  1.,  4.,  5.,  0.,  1.,  4.,  6.,
         1.,  8.,  1.,  1.,  4.,  3.,  1.,  1.,  1.,  6.,  0.,  5.,  1., 11.,
         4.,  1.,  1.,  7.,  3.,  6.,  9., 11.,  1.,  5.,  0.,  7.,  0.,  1.,
        11.,  6.,  1.,  5.,  1.,  0.,  7.,  9.,  7., 10.,  3.,  6.,  2., 10.,
         9.,  7.,  0.,  1.,  0.,  5., 10.,  7.,  3.,  0.,  1.,  1., 10.,  0.,
         9.,  5.,  7., 10.,  2.,  7.,  3.,  2., 10.,  2.,  2.,  1.,  9., 10.,
         2.,  7.,  5.,  7.,  3.,  6., 10.,  7.,  3.,  7.,  9.,  7.,  6., 10.,
         6.,  6., 10.,  7.,  9.,  7.,  6., 10.,  2.,  4., 10.,  5.,  8.,  1.,
         7.,  7.,  6.,  3.,  8.,  6., 10.,  1.,  6.,  8.,  7., 10.,  7.,  6.,
         4.,  5., 10.,  7.,  8.,  6.,  8., 10.,  5.,  7.,  7.,  4.,  4., 10.,
         7.,  8.,  5.,  6., 10.,  1.,  8.,  4.,  7.,  5.,  4.,  4.,  7.,  6.,
        10.,  8.,  5.,  6.,  8.,  4.,  8., 10.,  4.,  4.,  2.,  7.,  5.,  5.,
        10.,  3.,  8.,  5.,  4., 10.,  4.,  1.,  4.,  8.,  4., 10.,  5.,  2.,
        10.,  0.,  4.,  4.,  3.,  5.,  6.,  1.,  0.,  4.,  0.,  3.,  5.,  5.,
         1.,  0.,  0.,  4.,  4.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
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
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  5.,  0.,  4.,  0.,  4.,  0.,
         2.,  6.,  1.,  4.,  0., 10.,  6.,  1.,  1.,  5.,  0.,  7.,  1.,  4.,
         5.,  7.,  0.,  1.,  0., 10.,  5.,  3.,  4.,  1.,  6.,  0.,  3.,  7.,
         4., 10.,  0., 10.,  4.,  7.,  0.,  2.,  7.,  0.,  3.,  5.,  0., 10.,
         7.,  9.,  6.,  1., 10.,  7.,  3.,  3., 10.,  7.,  1.,  7., 10.,  6.,
         3.,  7., 10.,  1., 10.,  2.,  2.,  7.,  5.,  7., 10.,  2.,  3.,  2.,
         4.,  7.,  8., 10.,  2.,  3.,  4.,  7.,  2., 10.,  7.,  2.,  4.,  3.,
         7.,  2.,  9.,  7.,  3.,  0.,  5.,  7.,  2., 10.,  7.,  9.,  1.,  3.,
         6., 10.,  7.], device='cuda:0')
    '''
    observer_num_matrix = torch.matmul(visible_frames, visible_frames.transpose(0,1)) # torch.Size([661, 661]); 记录的是mask之间的observer number
    observer_num_list = observer_num_matrix.flatten() 
    observer_num_list = observer_num_list[observer_num_list > 0].cpu().numpy() # len: 159529; 记录的是mask之间的observer number
    observer_num_thresholds = [] # len:17 ; 记录的是mask之间的observer number的阈值; [24.0, 19.0, 16.0, 13.0, 12.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 4.0, 3.0, 3.0, 2.0, 2.0]
    for percentile in range(95, -5, -5):
        observer_num = np.percentile(observer_num_list, percentile)
        if observer_num <= 1:
            if percentile < 50:
                break
            else:
                observer_num = 1
        observer_num_thresholds.append(observer_num)

    # print("get_observer_num_thresholds: 4 embed")
    # embed()
    return observer_num_thresholds

def process_one_mask(point_in_mask_matrix, boundary_points, mask_point_cloud, frame_list, global_frame_mask_list, args):
    '''
        For a mask, compute the frames that it is visible and the masks that contains it.
    '''
    # mask_point_cloud 其实就是存的一个mask对于的各个point id
    # frame_list存的其实就是frame id，长度是83
    # visible_frame: torch.Size([83]), 长度是frame的个数，判断mask在某个frame当中是否可见
    # contained_mask: torch.Size([661]), 长度是 mask 的个数，判断一个frameid_maskid是否包含了该mask
    visible_frame = torch.zeros(len(frame_list)) # torch.Size([83]), 内容就是0/1组成
    contained_mask = torch.zeros(len(global_frame_mask_list)) # torch.Size([661]), 内容就是0/1组成

    # 一个mask出去边界得到的真正拥有的点id
    valid_mask_point_cloud = mask_point_cloud - boundary_points # len: 23; 内容：{87424, 87425, 87302, 88585, 88335, 87195, 86557, 87970, 85557, 87103, 88514, 85958, 88008, 87629, 86866, 88160, 89056, 89057, 88937, 86254, 88179, 85620, 86779}
    # 一个mask当中的点（这个比如有23个点），在不同的frame当中的哪个mask当中
    mask_point_cloud_info = point_in_mask_matrix[list(valid_mask_point_cloud), :] # shape: (23, 83); 0 means invisible，存储point在frame当中的哪个mask当中, 0表示不可见, 2表示在mask2中
    
    # 这里求的是对于23个点，我去看每一列也就是83列当中，每一列的和是否大于0，大于0，说明了这个mask当中的点会出现在这个frame当中，过滤得到所有可能的frame
    # 简单来说就是，一个这个mask出现在哪几个frame当中
    possibly_visible_frames = np.where(np.sum(mask_point_cloud_info, axis=0) > 0)[0] # len: 19; [ 0  1  6  7  8  9 10 11 12 13 14 15 16 17 72 74 75 76 77]; 

    split_num = 0 # 1
    visible_num = 0 # 13
    
    for frame_id in possibly_visible_frames:
        # mask_point_cloud_info[:, frame_id]: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1], len:21
        # mask_id_count: [ 0 21]
        mask_id_count = np.bincount(mask_point_cloud_info[:, frame_id]) # 存的是在一个不同mask_id出现的一个频次
        invisible_ratio = mask_id_count[0] / np.sum(mask_id_count) # 0 means that this point is invisible in this frame
        # If in a frame, most points in this mask are missing, then we think this mask is invisible in this frame.
        if 1 - invisible_ratio < args.mask_visible_threshold and (np.sum(mask_id_count) - mask_id_count[0]) < 500:
            continue
        visible_num += 1
        mask_id_count[0] = 0 # 将第一个元素置为0
        max_mask_id = np.argmax(mask_id_count) # 取出一个mask对应的所有point在当前frame内，出现次数最多的mask_id
        contained_ratio = mask_id_count[max_mask_id] / np.sum(mask_id_count)
        if contained_ratio > args.contained_threshold:
            visible_frame[frame_id] = 1 # 这个代表这个mask 在这个frame id当中是可见的
            frame_mask_idx = global_frame_mask_list.index((frame_list[frame_id], max_mask_id))
            contained_mask[frame_mask_idx] = 1
        else:
            split_num += 1 # This mask is splitted into two masks in this frame

    
    if visible_num == 0 or split_num / visible_num > args.undersegment_filter_threshold:
        return False, visible_frame, contained_mask
    else:
        return True, visible_frame, contained_mask

def process_masks(frame_list, global_frame_mask_list, point_in_mask_matrix, boundary_points, mask_point_clouds, args):
    '''
        For each mask, compute the frames that it is visible and the masks that contains it. 
        Meanwhile, we judge whether this mask is undersegmented.
    '''
    if args.debug:
        print('start processing masks')
    visible_frames = [] # len: 661; torch.Size([661, 83]); 0 means invisible, 1 means visible, 内容就是0/1
    contained_masks = [] # len: 661; torch.Size([661, 661]); 0 means not contained, 1 means contained, 内容就是0/1
    undersegment_mask_ids = [] # len: 30; [3, 10, 14, 19, 20, 24, 41, 58, 66, 77, 79, 214, 232, 242, 408, 413, 414, 446, 448, 492, 503, 527, 534, 546, 559, 560, 576, 592, 648, 656]; 记录的是mask在global_frame_mask_list中的index

    iterator = tqdm(global_frame_mask_list) if args.debug else global_frame_mask_list # len: 661
    for frame_id, mask_id in iterator:
        # valid： true/false，看mask id在frame id当中是否存在
        # visible_frame：判断当前mask在某个frame当中是否可见
        # contained mask：判断frameid maskid是否包含该mask
        valid, visible_frame, contained_mask = process_one_mask(point_in_mask_matrix, boundary_points, mask_point_clouds[f'{frame_id}_{mask_id}'], frame_list, global_frame_mask_list, args)
        visible_frames.append(visible_frame)
        contained_masks.append(contained_mask)
        if not valid:
            global_mask_id = global_frame_mask_list.index((frame_id, mask_id))
            undersegment_mask_ids.append(global_mask_id)

    visible_frames = torch.stack(visible_frames, dim=0).cuda() # (mask_num, frame_num), 0 means invisible, 1 means visible
    contained_masks = torch.stack(contained_masks, dim=0).cuda() # (mask_num, mask_num)

    # Undo the effect of undersegment observer masks to avoid merging two objects that are actually separated
    for global_mask_id in undersegment_mask_ids:
        frame_id, _ = global_frame_mask_list[global_mask_id]
        global_frame_id = frame_list.index(frame_id)
        mask_projected_idx = torch.where(contained_masks[:, global_mask_id])[0]
        contained_masks[:, global_mask_id] = False
        visible_frames[mask_projected_idx, global_frame_id] = False

    # print("process_masks: 3 embed")
    # embed()

    return visible_frames, contained_masks, undersegment_mask_ids