import numpy as np
import os
import torch
from utils.geometry import judge_bbox_overlay
from IPython import embed

def merge_overlapping_objects(total_point_ids_list, total_bbox_list, total_mask_list, overlapping_ratio):
    '''
        Merge objects that have larger than 0.8 overlapping ratio.
    '''
    total_object_num = len(total_point_ids_list)
    invalid_object = np.zeros(total_object_num, dtype=bool)

    for i in range(total_object_num):
        if invalid_object[i]:
            continue
        point_ids_i = set(total_point_ids_list[i])
        bbox_i = total_bbox_list[i]
        for j in range(i+1, total_object_num):
            if invalid_object[j]:
                continue
            point_ids_j = set(total_point_ids_list[j])
            bbox_j = total_bbox_list[j]
            if judge_bbox_overlay(bbox_i, bbox_j):
                intersect = len(point_ids_i.intersection(point_ids_j))
                if intersect / len(point_ids_i) > overlapping_ratio:
                    invalid_object[i] = True
                elif intersect / len(point_ids_j) > overlapping_ratio:
                    invalid_object[j] = True

    valid_point_ids_list = []
    valid_pcld_mask_list = []
    for i in range(total_object_num):
        if not invalid_object[i]:
            valid_point_ids_list.append(total_point_ids_list[i])
            valid_pcld_mask_list.append(total_mask_list[i])
    return valid_point_ids_list, valid_pcld_mask_list


def filter_point(point_frame_matrix, node, pcld_list, point_ids_list, mask_point_clouds, frame_list, args):
    '''
        Following OVIR-3D, we filter the points that hardly appear in this cluster (node), i.e. the detection ratio is lower than a threshold.
        Specifically, detection ratio = #frames that the point appears in this cluster (node) / #frames that the point appears in the whole video.
    '''
    def count_point_appears_in_video(point_frame_matrix, point_ids_list, node_global_frame_id_list):
        '''
            For all points in the cluster, compute #frames that the point appears in the whole video.
            Initialize #frames that the point appears in this cluster as 0.
        '''
        point_appear_in_video_nums, point_appear_in_node_matrixs = [], []
        for point_ids in point_ids_list:
            point_appear_in_video_matrix = point_frame_matrix[point_ids, ]
            point_appear_in_video_matrix = point_appear_in_video_matrix[:, node_global_frame_id_list]
            point_appear_in_video_nums.append(np.sum(point_appear_in_video_matrix, axis=1))
            
            point_appear_in_node_matrix = np.zeros_like(point_appear_in_video_matrix, dtype=bool) # initialize as False
            point_appear_in_node_matrixs.append(point_appear_in_node_matrix)
        return point_appear_in_video_nums, point_appear_in_node_matrixs

    def count_point_appears_in_node(mask_list, node_frame_id_list, point_ids_list, mask_point_clouds, point_appear_in_node_matrixs):
        '''
            Fillin the point_appear_in_node_matrixs by iterating the masks in this cluster (node).
            Meanwhile, since we split the disconnected point cloud into different objects, we also decide which object this mask belongs to.
            Besides, for each mask, we compute the coverage of this mask of the object it belongs to for furture use in OpenMask3D.
        '''
        object_mask_list = [[] for _ in range(len(point_ids_list))]

        for frame_id, mask_id in mask_list:
            frame_id_in_list = np.where(node_frame_id_list == frame_id)[0][0]
            mask_point_ids = list(mask_point_clouds[f'{frame_id}_{mask_id}'])

            object_id_with_largest_intersect, largest_intersect, coverage = -1, 0, 0
            for i, point_ids in enumerate(point_ids_list):
                point_ids_within_object = np.where(np.isin(point_ids, mask_point_ids))[0]
                point_appear_in_node_matrixs[i][point_ids_within_object, frame_id_in_list] = True
                if len(point_ids_within_object) > largest_intersect:
                    object_id_with_largest_intersect, largest_intersect = i, len(point_ids_within_object)
                    coverage = len(point_ids_within_object) / len(point_ids)
            if largest_intersect == 0:
                continue
            object_mask_list[object_id_with_largest_intersect] += [(frame_id, mask_id, coverage)]
        return object_mask_list, point_appear_in_node_matrixs

    node_global_frame_id_list = torch.where(node.visible_frame)[0].cpu().numpy()
    node_frame_id_list = np.array(frame_list)[node_global_frame_id_list]
    mask_list = node.mask_list
    
    point_appear_in_video_nums, point_appear_in_node_matrixs = count_point_appears_in_video(point_frame_matrix, point_ids_list, node_global_frame_id_list)
    object_mask_list, point_appear_in_node_matrixs = count_point_appears_in_node(mask_list, node_frame_id_list, point_ids_list, mask_point_clouds, point_appear_in_node_matrixs)

    # filter points
    filtered_point_ids, filtered_mask_list, filtered_bbox_list = [], [], []
    for i, (point_appear_in_video_num, point_appear_in_node_matrix) in enumerate(zip(point_appear_in_video_nums, point_appear_in_node_matrixs)):
        detection_ratio = np.sum(point_appear_in_node_matrix, axis=1) / (point_appear_in_video_num + 1e-6)
        valid_point_ids = np.where(detection_ratio > args.point_filter_threshold)[0]
        if len(valid_point_ids) == 0 or len(object_mask_list[i]) < 2:
            continue
        filtered_point_ids.append(point_ids_list[i][valid_point_ids])
        filtered_bbox_list.append([np.amin(pcld_list[i].points, axis=0), np.amax(pcld_list[i].points, axis=0)])
        filtered_mask_list.append(object_mask_list[i])
    return filtered_point_ids, filtered_bbox_list, filtered_mask_list


def dbscan_process(pcld, point_ids, DBSCAN_THRESHOLD=0.1):
    '''
        Following OVIR-3D, we use DBSCAN to split the disconnected point cloud into different objects.
    '''
    
    labels = np.array(pcld.cluster_dbscan(eps=DBSCAN_THRESHOLD, min_points=4)) + 1 # -1 for noise
    count = np.bincount(labels)

    # split disconnected point cloud into different objects
    pcld_list, point_ids_list = [], []
    pcld_ids_list = np.array(point_ids)
    for i in range(len(count)):
        remain_index = np.where(labels == i)[0]
        if len(remain_index) == 0:
            continue
        # new_pcld = pcld.select_by_index(remain_index)
        new_pcld = pcld.select_down_sample(remain_index)
        point_ids = pcld_ids_list[remain_index]
        pcld_list.append(new_pcld)
        point_ids_list.append(point_ids)
    return pcld_list, point_ids_list


def find_represent_mask(mask_info_list):
    mask_info_list.sort(key=lambda x: x[2], reverse=True)
    return mask_info_list[:5]


def export_class_agnostic_mask(args, class_agnostic_mask_list):
    pred_dir = os.path.join('data/prediction', args.config)
    os.makedirs(pred_dir, exist_ok=True)

    num_instance = len(class_agnostic_mask_list)
    pred_masks = np.stack(class_agnostic_mask_list, axis=1)
    pred_dict = {
        "pred_masks": pred_masks, 
        "pred_score":  np.ones(num_instance),
        "pred_classes" : np.zeros(num_instance, dtype=np.int32)
    }
    class_agnostic_pred_dir = os.path.join('data/prediction', args.config + '_class_agnostic')
    os.makedirs(class_agnostic_pred_dir, exist_ok=True)
    np.savez(os.path.join(class_agnostic_pred_dir, f'{args.seq_name}.npz'), **pred_dict)
    return


def export(dataset, total_point_ids_list, total_mask_list, args):
    '''
        Export class agnostic masks in standard evaluation format 
        and object dict with corresponding mask lists for semantic instance segmentation.
        Node that after clustering, a node = a cluster of masks = an object.
    '''
    total_point_num = dataset.get_scene_points().shape[0]
    class_agnostic_mask_list = []
    object_dict = {}
    for i, (point_ids, mask_list) in enumerate(zip(total_point_ids_list, total_mask_list)):
        object_dict[i] = {
            'point_ids': point_ids,
            'mask_list': mask_list,
            'repre_mask_list': find_represent_mask(mask_list),
        }
        binary_mask = np.zeros(total_point_num, dtype=bool)
        binary_mask[list(point_ids)] = True
        class_agnostic_mask_list.append(binary_mask)

    export_class_agnostic_mask(args, class_agnostic_mask_list)

    os.makedirs(os.path.join(dataset.object_dict_dir, args.config), exist_ok=True)
    np.save(os.path.join(dataset.object_dict_dir, args.config, 'object_dict.npy'), object_dict, allow_pickle=True)


def post_process(dataset, node_list, mask_point_clouds, scene_points, point_frame_matrix, frame_list, args):
    if args.debug:
        print('start exporting')
    
    
    # For each cluster, we follow OVIR-3D to i) use DBScan to split the disconnected point cloud into different objects
    # ii) filter the points that hardly appear within this cluster, i.e. the detection ratio is lower than a threshold
    total_point_ids_list, total_bbox_list, total_mask_list = [], [], []
    '''
    total_bbox_list: len 39
    total_bbox_list[0]
    [array([3.86787105, 0.55813628, 0.67466122]), array([4.58438778, 1.22155011, 0.86303276])]
    '''
    for node in (node_list):
        if len(node.mask_list) < 2: # objects merged from less than 2 masks are ignored
            continue
        
        pcld, point_ids = node.get_point_cloud(scene_points)
        pcld_list, point_ids_list = dbscan_process(pcld, point_ids) # split the disconnected point cloud into different objects
        
        point_ids_list, bbox_list, mask_list = filter_point(point_frame_matrix, node, pcld_list, point_ids_list, mask_point_clouds, frame_list, args)

        total_point_ids_list.extend(point_ids_list)
        total_bbox_list.extend(bbox_list)
        total_mask_list.extend(mask_list)

    # merge objects that have larger than 0.8 overlapping ratio
    '''
    total_point_ids_list: len 39
    total_point_ids_list[0]
    [ 88064  88065  88066  88067  86020  86021  88071  88072  86030  88082
    88084  86037  88085  88086  86055  88106  88111  86069  88118  88119
    88120  88122  88123  88124  88125  88126  88127  88132  88133  86094
    86095  88160  88164  88175  88176  88177 114802  88179  88178  86132
    86133  86134  86135  86136  86137  88180  88181  88182  88183  88184
    88185  86146  86147  88195  88201  88202  86159  88211  88212  88214
    88215  88216  88217  88218  88230  88231  88232  88233  86186  88234
    88235  88236  88249  88251  88257  86211  86212  86213  86217  86218
    86219  86221  86222  88270  88271  88272  88273  88284  88285  88286
    88287  88288  88289  88290  88291  86251  86252  86253  86254  86255
    86256  86257  86258  86259  88307  88308  88315  88316  88317  88320
    88321  88322  88323  86278  86279  86280  86281  86282  86283  86284
    88335  88336  88340  86293  88365  88366  88368  88369  88370  88371
    88372  86322  86323  86324  86325  88382  88383  88384  86337  86338
    86339  88385  88403  88404  88405  88406  88410  88416  86369  88417
    88422  88423  88424  86377  86378  88425  88426  88427  88428  88431
    88435  88442  86411  86412  88461  88462  88463  86419  88473  88474
    88475  88476  88477  86431  88483  86438  86439  88487  88488  86444
    88489  88490  88491  88496  88499  88500  88504  88505  86462  86463
    88514  88517  88525  88526  88528  86484  86485  86486  86487  86488
    88537  88538  88539  88547  88548  88549  88550  88551  88556  88557
    88562  88563  88564  88570  86526  86527  86529  86530  86531  86532
    88583  88584  88585  88586  88587  88594  88595  86557  86558  86562
    88612  86565 115238  88617  86577  88627  88628  88629  88633  88634
    88635  88636  86590  86591  86592  86593  86594  86595  86596  86597
    86598  88643  86606  86607  86608  86610  86611  86612  86613  88663
    88664  86624  86629  86630  88678  88679  86633  86634  86635  88680
    88681  88686  88688  88689  88690  86655  88712  88713  86667  86668
    86669  88728  88729  86682  88730  88740  88743  88744  88746  88747
    88751  86707  86708  86709  86710 115381  88766  86719  88767  88768
    86726  86727  86728  86729  88775  86740  86741  86747  86748  88798
    86756  86757  86758  88808  88809  88810  88811  88812  88821  86777
    86778  86780  86781  86785  86786  86787  86788  88844  88845  86800
    86801  86802  88852  88853  86806  86807  86809  86810  86811  88860
    88861  88862  86820  88875  86831  86832 115504 115505  88885  88898
    88903  86856  88904  86860  88910  88911  88912  86865  86866  88913
    86868  88921  86876  86877  86881  86882  86883  86884  86885  86886
    86887  86888  88937  86889  86890  88938  88939  88951  88952  88953
    88954  88955  88957  88959  88963  88964  86918  86925  86926  86927
    86933  86934  86935  86936  86937  86938  86939  86940  86941  86942
    88985  88986  86947  86948  88999  89000  86955  89011  89012  89013
    86984  86988  86989  86990  86991  86992  89036  89037  89038  89039
    89043  87000  89044  89045  89046  89047  87005  89056  89057  89058
    87021  87022  89078  89083  89086  89087  87043  87044  87045  87046
    87047  89105  89106  89107  87060  89108  89109  89110  89111  87073
    87093  87094  87095  87096  87102  87103  87125  87126  87128  87130
    87131  87143  87144  87195  87196  87197  87216  85721  85722  85723
    87250  87251  87252  87253  87254  87255  87256  87257  87263  87264
    87266  87271  87272  87273  87274  87275  87276  87277  87278  87279
    87280  87281  87282  87301  87302 113931 113932  87340  87349  87350
    87351  87352  87353  87354  87363  87364  87365  87366  87367  87368
    87372  87373  87374  87375  87376  87377  87378  87392  87393  87394
    87396  87398  87399  87400  87401  87402  87407  87409  87410  87411
    87421  87422  87424  87425  87460  87461  87462  87463  87469  87473
    114104  87486  87487  87488  87490 114139 114140  87523  87524  87525
    87526  87527  87528  87539 114167 114169  87556  87557  85523  85524
    85528  85529  85534  85535  85536  87591  87592  87593  87597  87598
    87602  87603  87604  85557  85556  85558  87608  87614  87615  87616
    87625  87626  87629  85584  85586  87636  87637  87638  87639  87640
    87641  87642  85611  85612  85613  85614  85615  85619  85620  85629
    87680  87681  85633  85634  85635  85636  85637  85643  85644  85645
    85646  85647  87696  87697  87698  85651  85658  85659  85660  85661
    85662  85663  85664  85665  85666  85667  85668  87718  85669  85672
    87720  87721  87722  85670  85671  87729  87732  87733  85684  87743
    85696  85697  87744  85695  87746  87747  87748  87749  87750  87751
    87752  87753  87754  87755  85708  85709  85710  87760  85719  85720
    87766  87767  87768  87769  87770  87771  87772  87773  87774  87775
    87776  87777  85724  85725  85726  85727  85737  85738  87787  85739
    87792  87793  87799  85765  87815  87818  85770  85772  85773  85774
    85782  87843  87844  87851  87852  87855  87856  87861  87862  87863
    85821  85824  85825  87872  87873  87879  87880  87881  87882  85830
    85842  85844  85845  87894  87896  87897  87898  87901  87902  87903
    87908  87909  85877  85886  87939  87940  87945  87958  87959  85913
    87968  87970  87971  85940  85941  87991  87992  88005  85958  88006
    88008  88007  88010  88014  85959  85962  88024  88032  88033  88047
    86012  88063]
    '''
    total_point_ids_list, total_mask_list = merge_overlapping_objects(total_point_ids_list, total_bbox_list, total_mask_list, overlapping_ratio=0.8)
    '''
    total_mask_list: len 39
    total_mask_list[0]
    [(0, 1, 0.04423748544819558), (750, 5, 0.050058207217694994), (170, 6, 0.49941792782305006), (120, 5, 0.7089639115250291), (760, 5, 0.5273573923166472), (70, 4, 0.5075669383003493), (770, 4, 0.42724097788125726), (160, 8, 0.7543655413271245), (130, 3, 0.7532013969732246), (700, 2, 0.0640279394644936), (80, 2, 0.6076833527357393), (150, 6, 0.6169965075669382), (90, 2, 0.7031431897555297), (140, 3, 0.6984866123399301), (720, 6, 0.05471478463329453), (100, 3, 0.6693830034924331), (740, 4, 0.1979045401629802), (110, 3, 0.7054714784633295), (750, 3, 0.30384167636786963), (60, 4, 0.3050058207217695), (820, 1, 0.01979045401629802), (20, 1, 0.03841676367869616)]
    '''
    # print("post_process: 10 embed")
    # embed()
    export(dataset, total_point_ids_list, total_mask_list, args)
    return