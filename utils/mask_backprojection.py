import numpy as np
from pytorch3d.ops import ball_query
import torch
import open3d as o3d
from utils.geometry import denoise
from torch.nn.utils.rnn import pad_sequence
from IPython import embed

COVERAGE_THRESHOLD = 0.3
DISTANCE_THRESHOLD = 0.03
FEW_POINTS_THRESHOLD = 25
DEPTH_TRUNC = 20
BBOX_EXPAND = 0.1


def backproject(depth, intrinisc_cam_parameters, extrinsics):
    """
    convert color and depth to view pointcloud
    """
    depth = o3d.geometry.Image(depth)
    pcld = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinisc_cam_parameters, depth_scale=1, depth_trunc=DEPTH_TRUNC)
    pcld.transform(extrinsics)
    return pcld


def get_neighbor(valid_points, scene_points, lengths_1, lengths_2):
    _, neighbor_in_scene_pcld, _ = ball_query(valid_points, scene_points, lengths_1, lengths_2, K=20, radius=DISTANCE_THRESHOLD, return_nn=False)
    return neighbor_in_scene_pcld


def get_depth_mask(depth):
    depth_tensor = torch.from_numpy(depth).cuda()
    depth_mask = torch.logical_and(depth_tensor > 0, depth_tensor < DEPTH_TRUNC).reshape(-1)
    return depth_mask


def crop_scene_points(mask_points, scene_points):
    x_min, x_max = torch.min(mask_points[:, 0]), torch.max(mask_points[:, 0])
    y_min, y_max = torch.min(mask_points[:, 1]), torch.max(mask_points[:, 1])
    z_min, z_max = torch.min(mask_points[:, 2]), torch.max(mask_points[:, 2])

    selected_point_mask = (scene_points[:, 0] > x_min) & (scene_points[:, 0] < x_max) & (scene_points[:, 1] > y_min) & (scene_points[:, 1] < y_max) & (scene_points[:, 2] > z_min) & (scene_points[:, 2] < z_max)
    selected_point_ids = torch.where(selected_point_mask)[0]
    cropped_scene_points = scene_points[selected_point_ids]
    return cropped_scene_points, selected_point_ids


def turn_mask_to_point(dataset, scene_points, mask_image, frame_id): # 针对一个frame操作
    '''
    外参： 表示从世界坐标到相机坐标的变换
    [[-0.8691   -0.195987  0.454153  0.916851]
     [-0.485097  0.158265 -0.860019  4.065093]
     [ 0.096676 -0.967751 -0.232621  0.829043]
     [ 0.        0.        0.        1.      ]]
    前三行包括旋转矩阵和平移向量，最后一行用于齐次坐标。
    '''
    intrinisc_cam_parameters = dataset.get_intrinsics(frame_id) # 相机内参
    extrinsics = dataset.get_extrinsic(frame_id) # 相机外参
    if np.sum(np.isinf(extrinsics)) > 0: # 判断外参是否有效
        return {}, [], set()

    mask_image = torch.from_numpy(mask_image).cuda().reshape(-1) # torch.size([307200]): [9,9,9...,13,13,13]
    ids = torch.unique(mask_image).cpu().numpy() # 展出所有唯一的mask id, 
    ids.sort()# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13], len:14
    
    depth = dataset.get_depth(frame_id) # depth: len 480; type: numpy.ndarray; shape: (480, 640); depth[0]:[0,0,...,2.442,2.422,...,0,..,0]
    depth_mask = get_depth_mask(depth) # 获取深度图，并根据深度图计算有效深度的mask。torch.Size([307200]):[False, ..., True, ...]

    colored_pcld = backproject(depth, intrinisc_cam_parameters, extrinsics) # geometry::PointCloud with 286532 points;<class 'open3d.open3d.geometry.PointCloud'>
    view_points = np.asarray(colored_pcld.points) # 使用深度图和相机参数进行反向投影，计算3D点云。shape: (286532, 3), view_points[0]: [1.32382853 1.30404534 1.33365807]

    mask_points_list = [] # len: 11; mask_points_list[0].shape: torch.Size([34, 3]); 一个mask对应了34个点, 也可能是435，取决于这个mask大小
    mask_points_num_list = [] # [34, 435, 998, 61, 331, 1001, 941, 105, 1288, 461, 4449]
    scene_points_list = [] # len: 11; scene_points_list[0].shape: torch.Size([892, 3]);
    scene_points_num_list = [] # [892, 1128, 3818, 168, 287, 2240, 3503, 6970, 5712, 614, 14825]
    selected_point_ids_list = [] # len: 11; selected_point_ids_list[0].shape: torch.Size([892]), tensor([ 85528,  85529,..., 115766]) 代表了mask对应的点在scene中的ID
    initial_valid_mask_ids = [] # len: 11;  内容：[1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
    for mask_id in ids:
        if mask_id == 0:
            continue
        segmentation = mask_image == mask_id
        valid_mask = segmentation[depth_mask].cpu().numpy()

        mask_pcld = o3d.geometry.PointCloud()
        mask_points = view_points[valid_mask]
        if len(mask_points) < FEW_POINTS_THRESHOLD:
            continue
        mask_pcld.points = o3d.utility.Vector3dVector(mask_points)

        mask_pcld = mask_pcld.voxel_down_sample(voxel_size=DISTANCE_THRESHOLD)
        mask_pcld, _ = denoise(mask_pcld)
        mask_points = np.asarray(mask_pcld.points)
        
        if len(mask_points) < FEW_POINTS_THRESHOLD:
            continue
        
        mask_points = torch.tensor(mask_points).float().cuda()
        cropped_scene_points, selected_point_ids = crop_scene_points(mask_points, scene_points)
        initial_valid_mask_ids.append(mask_id)
        mask_points_list.append(mask_points)
        scene_points_list.append(cropped_scene_points)
        mask_points_num_list.append(len(mask_points))
        scene_points_num_list.append(len(cropped_scene_points))
        selected_point_ids_list.append(selected_point_ids)

    if len(initial_valid_mask_ids) == 0:
        return {}, [], []
    mask_points_tensor = pad_sequence(mask_points_list, batch_first=True, padding_value=0) # shape: torch.Size([11, 4365, 3])
    scene_points_tensor = pad_sequence(scene_points_list, batch_first=True, padding_value=0) # shape: torch.Size([11, 14427, 3])

    lengths_1 = torch.tensor(mask_points_num_list).cuda() # len: 11; tensor([  34,  443,  991,   63,  333, 1000,  949,  105, 1273,  461, 4365]
    lengths_2 = torch.tensor(scene_points_num_list).cuda() # len: 11; tensor([  892,  1135,  3686,   180,   287,  2199,  3557,  6970,  5504,   614, 14427]
    '''
    neighbor_in_scene_pcld[0][:3]
    tensor([[209,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
          -1,  -1,  -1,  -1,  -1,  -1],
            [466,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
          -1,  -1,  -1,  -1,  -1,  -1],
            [ 59, 347, 350,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
          -1,  -1,  -1,  -1,  -1,  -1]], device='cuda:0')
    '''
    neighbor_in_scene_pcld = get_neighbor(mask_points_tensor, scene_points_tensor, lengths_1, lengths_2) # shape: torch.Size([11, 4365, 20])

    valid_mask_ids = [] # 存储有效的mask id; len: 9; [1, 3, 4, 5, 6, 7, 8, 11, 13]
    '''
    mask_info[1]
    {87424, 87425, 87301, 87302, 88585, 88587, 88335, 85651, 87195, 86557, 87970, 88612, 88106, 88369, 88370, 88371, 88372, 85557, 88373, 87103, 88514, 85958, 86984, 88008, 88010, 87629, 86866, 88160, 89056, 89057, 88937, 86254, 88179, 85620, 86777, 86778, 86779, 88318}
    上面的这个就是点的id
    '''
    mask_info = {} # key：maskid；value：mask对应在scene当中的point id集合; len: 9; dict_keys([1, 3, 4, 5, 6, 7, 8, 11, 13]); dict_values(len: 38 891 1848 123 257 1478 622 1808 5803 )
    frame_point_ids = set() # 用来存储所有有效掩码对应的点ID; len: 12416, 存所有的点ID

    for i, mask_id in enumerate(initial_valid_mask_ids):
        mask_neighbor = neighbor_in_scene_pcld[i] # P, 20
        mask_point_num = mask_points_num_list[i] # Pi
        mask_neighbor = mask_neighbor[:mask_point_num] # Pi, 20

        valid_neighbor = mask_neighbor != -1 # Pi, 20
        neighbor = torch.unique(mask_neighbor[valid_neighbor])
        neighbor_in_complete_scene_points = selected_point_ids_list[i][neighbor].cpu().numpy()
        coverage = torch.any(valid_neighbor, dim=1).sum().item() / mask_point_num

        if coverage < COVERAGE_THRESHOLD:
            continue
        valid_mask_ids.append(mask_id)
        mask_info[mask_id] = set(neighbor_in_complete_scene_points)
        frame_point_ids.update(mask_info[mask_id])
    # print("turn_mask_to_point: 0 embed")
    # embed()

    return mask_info, valid_mask_ids, list(frame_point_ids)


def frame_backprojection(dataset, scene_points, frame_id):
    mask_image = dataset.get_segmentation(frame_id, align_with_depth=True)
    mask_info, _, frame_point_ids = turn_mask_to_point(dataset, scene_points, mask_image, frame_id)
    return mask_info, frame_point_ids