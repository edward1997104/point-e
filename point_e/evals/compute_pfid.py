from dataclasses import dataclass
import tyro
from point_e.evals.fid_is import compute_statistics
import torch
from point_e.models.download import load_checkpoint
from point_e.evals.pointnet2_cls_ssg import get_model
import numpy as np
import glob
import trimesh
import os
import math
from multiprocessing import Pool
import pandas as pd

def normalize_point_clouds(pc: np.ndarray) -> np.ndarray:
    centroids = np.mean(pc, axis=1, keepdims=True)
    pc = pc - centroids
    m = np.max(np.sqrt(np.sum(pc**2, axis=-1, keepdims=True)), axis=1, keepdims=True)
    pc = pc / m
    return pc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@dataclass
class Args:
    gt_folder : str
    pred_folder : str
    workers : int
    batch_size : int = 32
    use_aligned: bool = False
    cache_dir: str = './cache_pointnet'

args = tyro.cli(Args)

def sample_pointcloud(args):
    mesh_path, output_point_path, point_num = args
    mesh = trimesh.load(mesh_path)
    points, _ = trimesh.sample.sample_surface(mesh, point_num)
    points = normalize_point_clouds(points)
    np.save(output_point_path, points)
    print(f"{output_point_path} done...")
    return points


def load_pointnet():
    state_dict = load_checkpoint("pointnet", device=torch.device("cpu"), cache_dir=args.cache_dir)[
        "model_state_dict"
    ]
    model = get_model(num_class=40, normal_channel=False, width_mult=2)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model

def get_pointnet_feature(points, model, batch_size):
    output_features = []
    batch_cnt =  int(math.ceil(len(points) / batch_size))
    for batch_index in range(batch_cnt):
        batch_feature = model(points[batch_index * batch_size: (batch_index + 1) * batch_size], features=True)
        output_features.append(batch_feature[2].cpu().detach().numpy())
        print(f"batch {batch_index} done")
    output_features = np.concatenate(output_features, axis=0)
    return output_features


if __name__ == '__main__':
    gt_obj_paths = glob.glob(os.path.join(args.gt_folder, '*.obj'), recursive=False)

    item_to_process = []
    for gt_obj_path in gt_obj_paths:
        id = os.path.basename(gt_obj_path)[:-4]
        postfix = '_aligned.obj' if args.use_aligned else '.obj'
        pred_path = os.path.join(args.pred_folder, f'{id}{postfix}')
        if os.path.exists(pred_path):
            item_to_process.append((gt_obj_path, pred_path))

    print(f"Found {len(item_to_process)} items to process")

    if os.path.exists(f'{args.pred_folder}/pfid.csv'):
        print("iou.csv already exists, skipping")
        exit()

    # sampling point clouds
    print("Sampling point clouds...")
    args_inputs = []
    for item in item_to_process:
        gt_obj_path, pred_path = item
        id = os.path.basename(gt_obj_path)[:-4]
        postfix = '_aligned.obj' if args.use_aligned else '.obj'
        pred_path = os.path.join(args.pred_folder, f'{id}{postfix}')
        gt_point_path = os.path.join(args.gt_folder, f'{id}.npy')
        pred_point_path = os.path.join(args.pred_folder, f'{id}.npy')
        if not os.path.exists(gt_point_path):
            args_inputs.append((gt_obj_path, gt_point_path, 4096))
        if not os.path.exists(pred_point_path):
            args_inputs.append((pred_path, pred_point_path, 4096))

    pool = Pool(args.workers)
    pool.map(sample_pointcloud, args_inputs)

    result_gt_points, result_pred_points = [], []
    for item in item_to_process:
        gt_obj_path, pred_path = item
        id = os.path.basename(gt_obj_path)[:-4]
        postfix = '_aligned.obj' if args.use_aligned else '.obj'
        pred_path = os.path.join(args.pred_folder, f'{id}{postfix}')
        gt_point_path = os.path.join(args.gt_folder, f'{id}.npy')
        pred_point_path = os.path.join(args.pred_folder, f'{id}.npy')
        if not os.path.exists(gt_point_path):
            sample_pointcloud(gt_obj_path, gt_point_path, 4096)
        if not os.path.exists(pred_point_path):
            sample_pointcloud(pred_path, pred_point_path, 4096)

        # load_points
        gt_points = np.load(gt_point_path)
        pred_points = np.load(pred_point_path)

        # add to list
        result_gt_points.append(gt_points[None, :])
        result_pred_points.append(pred_points[None, :])

    print("Sampling done")

    # concat point clouds
    result_gt_points = np.concatenate(result_gt_points, axis=0)
    result_pred_points = np.concatenate(result_pred_points, axis=0)

    # convert to torch tensor
    result_gt_points = torch.from_numpy(result_gt_points).float().to(device)
    result_pred_points = torch.from_numpy(result_pred_points).float().to(device)

    # switch axis 1 2
    result_gt_points = result_gt_points.transpose(1, 2)
    result_pred_points = result_pred_points.transpose(1, 2)

    # obtain pointnet
    pointnet = load_pointnet()

    # get pointnet feature
    print("Extracting gt pointnet feature...")
    result_gt_features = get_pointnet_feature(result_gt_points, pointnet, args.batch_size)
    print("result_gt_features.shape", result_gt_features.shape)
    print("Extracting pred pointnet feature...")
    result_pred_features = get_pointnet_feature(result_pred_points, pointnet, args.batch_size)
    print("result_pred_features.shape", result_pred_features.shape)

    # compute pfid
    gt_stats = compute_statistics(result_gt_features)
    pred_stats = compute_statistics(result_pred_features)

    # compute pfid
    results = []
    avg_pfid = pred_stats.frechet_distance(gt_stats)
    print(f"P-FID: {avg_pfid}")

    results.append(('avg_pfid', avg_pfid))

    # save results
    df = pd.DataFrame(results, columns=['ID', 'Value'])
    df.to_csv(f'{args.pred_folder}/pfid.csv', index=False)

    print(f"Processed {len(item_to_process)} items")




