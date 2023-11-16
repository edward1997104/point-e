from PIL import Image
import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from point_e.util.pc_to_mesh import marching_cubes_mesh
import trimesh
import os
import numpy as np
import math
import boto3
import tempfile
import cloudpathlib
from dataclasses import dataclass
import tyro
import multiprocessing
import pickle

@dataclass
class Args:
    workers : int = 8
    output_dir : str  = 'pointe-mesh-output'
    bucket : str = 'gso-renders'
    ext : str = '018.png'
    cuda_cnt : int = 8
    file_path_pickle : str = 'file_paths.pkl'
    render_resolution : int = 384


args = tyro.cli(Args)

def scale_to_unit_cube(mesh, scale_ratio = 0.9):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)
    vertices *= scale_ratio

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

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

def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    worker_idx : int
) -> None:

    cuda_id = worker_idx % args.cuda_cnt
    torch.cuda.set_device(f'cuda:{cuda_id}')
    device = torch.device(f'cuda:{cuda_id}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)

    print('creating base model...')
    base_name = 'base40M'  # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    print('creating SDF model...')
    name = 'sdf'
    model = model_from_config(MODEL_CONFIGS[name], device)
    model.eval()

    print('loading SDF model...')
    model.load_state_dict(load_checkpoint(name, device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )

    while True:
        item = queue.get()
        if item is None:
            break
        try:
            process_one(item, model, sampler, cuda_id)
        except Exception as e:
            print(e)
        queue.task_done()
        with count.get_lock():
            count.value += 1

img_bucket_mapping = {
            f'ABC_renders_{args.render_resolution}': 'abc-renders',
            f'BuildingNet_renders_{args.render_resolution}': 'buildingnet-renders',
            f'Fusion_renders_{args.render_resolution}': 'fusion-renders',
            f'ModelNet40_renders_{args.render_resolution}': 'modelnet40-renders',
            f'Objaverse_renders_{args.render_resolution}': '000-objaverse-renders',
            f'ShapeNet_V2_renders_{args.render_resolution}': 'shapenet-v2-renders',
            f'Thingi10K_renders_{args.render_resolution}': 'thingi10k-renders',
            f'Thingiverse_renders_{args.render_resolution}': 'thingiverse-renders',
            f'Github_renders_{args.render_resolution}': 'github-renders',
            f'Infinigen_renders_{args.render_resolution}': 'infinigen-renders-us',
            f'Smpl_renders_{args.render_resolution}': 'smpl-renders',
            f'Smal_renders_{args.render_resolution}': 'smal-renders',
            f'Coma_renders_{args.render_resolution}': 'coma-renders',
            f'DeformingThings4D_renders_{args.render_resolution}': 'deformingthings4d-renders',
            f'Abo_renders_{args.render_resolution}': 'abo-renders',
            f'Fg3d_renders_{args.render_resolution}': 'fg3d-renders',
            f'House3d_renders_{args.render_resolution}': 'house3d-renders',
            f'Toy4k_renders_{args.render_resolution}': 'toy4k-renders',
            f'Gso_renders_{args.render_resolution}': 'gso-renders',
            f'3DFuture_renders_{args.render_resolution}': '3dfuture-renders',
        }

def process_one(img, model, sampler, cuda_id):

    dataset, id = img
    dataset = dataset.replace("_wavelet_latents", f'_renders_{args.render_resolution}').replace("_fixed", "")
    bucket = img_bucket_mapping[dataset]



    torch.cuda.set_device(f'cuda:{cuda_id}')
    device = torch.device(f'cuda:{cuda_id}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)

    obj_path = os.path.join(args.output_dir, f'{id}.obj')
    if os.path.exists(obj_path):
        print("File exists: ", obj_path)
        return



    with tempfile.TemporaryDirectory() as tmp_dir:
        print("start processing: ", img)
        cloudpath = cloudpathlib.CloudPath(f's3://{bucket}/{id}/img/{args.ext}')
        save_filename = f"{id}.png"
        save_path = os.path.join(tmp_dir, save_filename)
        cloudpath.download_to(save_path)

        img = Image.open(save_path)

        # Produce a sample from the model.
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
            samples = x

        pc = sampler.output_to_point_clouds(samples)[0]

        mesh = marching_cubes_mesh(
            pc=pc.coords,
            model=model,
            batch_size=4096,
            grid_size=128,  # increase to 128 for resolution used in evals
            progress=True,
            fill_vertex_channels=False,
        )
        mesh = trimesh.Trimesh(vertices=mesh.verts, faces=mesh.faces)
        mesh = rotate_around_axis(mesh, axis='x', reverse=False)

        mesh.export(obj_path)



if __name__ == '__main__':

    os.makedirs(args.output_dir, exist_ok=True)


    with open(args.file_path_pickle, 'rb') as f:
        img_lists = pickle.load(f)

    print("Number of images: ", len(img_lists))

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    for worker_i in range(args.workers):
        process = multiprocessing.Process(
            target=worker, args=(queue, count, worker_i)
        )
        process.daemon = True
        process.start()

    for img in img_lists:
        queue.put(img)

    queue.join()

    for _ in range(args.workers):
        queue.put(None)

    print(f'Processed {count.value} models')