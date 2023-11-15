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

if __name__ == '__main__':

    output_dir = 'pointe-mesh-output'
    os.makedirs(output_dir, exist_ok=True)

    bucket = 'gso-renders'
    ext = '018.png'
    s3 = boto3.resource('s3')

    ## getting file with extensions in s3 bucket
    bucket = s3.Bucket(bucket)
    img_lists = []
    for obj in bucket.objects.filter(Prefix=''):
        if obj.key.endswith(ext):
            img_lists.append(obj.key)

    print("Number of images: ", len(img_lists))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    with tempfile.TemporaryDirectory() as tmp_dir:
        for img in img_lists:
            print("start processing: ", img)
            cloudpath = cloudpathlib.CloudPath(f's3://gso-renders/{img}')
            id = img.split('/')[-3]
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
            mesh.export(os.path.join(output_dir, f'{id}.obj'))