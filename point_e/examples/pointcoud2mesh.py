from PIL import Image
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud
import trimesh
import numpy as np
import math
from dataclasses import dataclass
import glob
import tyro
import os

@dataclass
class Args:
    gt_folder : str
    output_folder : str
    point_num : int = 4096

args = tyro.cli(Args)

def rotate_around_axis(mesh, axis = 'x', reverse = False):
    if reverse:
        angle = math.pi / 2
    else:
        angle = -math.pi / 2

    if axis == 'x':
        direction = [1, 0, 0]
    elif axis == 'y':
        direction = [0, 1, 0]
    else:
        direction = [0, 0, 1]

    center = mesh.centroid

    rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)

    mesh.apply_transform(rot_matrix)

    return mesh

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('creating SDF model...')
    name = 'sdf'
    model = model_from_config(MODEL_CONFIGS[name], device)
    model.eval()

    print('loading SDF model...')
    model.load_state_dict(load_checkpoint(name, device))

    # Check meshes in gt_folder
    obj_paths = glob.glob(os.path.join(args.gt_folder, '*.obj'))

    for obj_path in obj_paths:
        id = os.path.basename(obj_path)[:-4]
        print("Processing : ", id)

        # Load point cloud
        mesh = trimesh.load(obj_path, force = 'mesh')
        pc = mesh.sample(args.point_num)

        # Produce a mesh (without vertex colors)
        mesh = marching_cubes_mesh(
            pc=pc,
            model=model,
            batch_size=4096,
            grid_size=128,  # increase to 128 for resolution used in evals
            progress=True,
            fill_vertex_channels=False,
        )
        mesh = trimesh.Trimesh(vertices=mesh.verts, faces=mesh.faces)
        mesh.export(os.path.join(args.output_folder, f'{id}.obj'))

        print("Saved : ", os.path.join(args.output_folder, f'{id}.obj'))