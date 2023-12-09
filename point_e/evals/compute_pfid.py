from dataclasses import dataclass
import tyro
from point_e.evals.fid_is import compute_statistics
import torch
from point_e.models.download import load_checkpoint

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True, width_mult=1):
        super(get_model, self).__init__()
        self.width_mult = width_mult
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=in_channel,
            mlp=[64 * width_mult, 64 * width_mult, 128 * width_mult],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 * width_mult + 3,
            mlp=[128 * width_mult, 128 * width_mult, 256 * width_mult],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 * width_mult + 3,
            mlp=[256 * width_mult, 512 * width_mult, 1024 * width_mult],
            group_all=True,
        )
        self.fc1 = nn.Linear(1024 * width_mult, 512 * width_mult)
        self.bn1 = nn.BatchNorm1d(512 * width_mult)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512 * width_mult, 256 * width_mult)
        self.bn2 = nn.BatchNorm1d(256 * width_mult)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256 * width_mult, num_class)

    def forward(self, xyz, features=False):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024 * self.width_mult)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        result_features = self.bn2(self.fc2(x))
        x = self.drop2(F.relu(result_features))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        if features:
            return x, l3_points, result_features
        else:
            return x, l3_points

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
    workers: int
    scale_ratio: float = 0.9
    batch_size = 256
    use_normalize: bool = True
    use_aligned: bool = False

args = tyro.cli(Args)

def sample_pointcloud(mesh_path, output_point_path, point_num):
    mesh = trimesh.load(mesh_path)
    points, _ = trimesh.sample.sample_surface(mesh, point_num)
    points = normalize_point_clouds(points)
    np.save(output_point_path, points)
    return points


def load_pointnet():
    state_dict = load_checkpoint("pointnet", device=torch.device("cpu"), cache_dir=cache_dir)[
        "model_state_dict"
    ]
    model = get_model(num_class=40, normal_channel=False, width_mult=2)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

def get_pointnet_feature(points,  model, batch_size):
    output_features = []
    batch_cnt =  int(math.ceil(len(points) / batch_size))
    for batch_index in range(batch_cnt):
        batch_feature = model(batch[batch_index * batch_size: (batch_index + 1) * batch_size])
        output_features.append(batch_feature.cpu().detach().numpy())
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

    # sampling point clouds
    print("Sampling point clouds...")
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
    print("Extracting pred pointnet feature...")
    result_pred_features = get_pointnet_feature(result_pred_points, pointnet, args.batch_size)

    # compute pfid
    gt_stats = compute_statistics(result_gt_features)
    pred_stats = compute_statistics(result_pred_features)

    # compute pfid
    print(f"P-FID: {gt_stats.frechet_distance(pred_stats)}")




