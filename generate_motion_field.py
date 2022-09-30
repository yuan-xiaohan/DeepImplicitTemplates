#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import numpy as np
import os
import torch
import sys
import deep_sdf
import deep_sdf.workspace as ws
import SimpleITK as sitk

def create_motion_field(
    decoder, latent_vec, filename, N=256, max_batch=(32 ** 3 * 4), offset=None, scale=None, volume_size=2.0
):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-volume_size/2.0, -volume_size/2.0, -volume_size/2.0]
    voxel_size = volume_size / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    warped = []
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
        latent_repeat = latent_vec.expand(sample_subset.shape[0], -1)
        inputs = torch.cat([latent_repeat, sample_subset], 1)
        with torch.no_grad():
            warped_, _ = decoder(inputs, output_warped_points=True)
        warped_ = warped_.detach().cpu().numpy()
        warped.append(warped_)
        head += max_batch

    warped = np.concatenate(warped, axis=0)
    warped = warped.reshape([N, N, N, 3])

    # Normalize values to [-1, 1]
    warped = (warped + 1) / 2

    return warped


def code_to_mesh(experiment_directory, checkpoint, output_dir,
                 keep_normalized=False, resolution=256):

    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, checkpoint + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    decoder.eval()

    clamping_function = None
    if specs["NetworkArch"] == "deep_sdf_decoder":
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])
    elif specs["NetworkArch"] == "deep_implicit_template_decoder":
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])

    latent_vectors = ws.load_pre_trained_latent_vectors(experiment_directory, checkpoint)
    latent_vectors = latent_vectors.cuda()

    train_split_file = specs["TrainSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    data_source = specs["DataSource"]

    instance_filenames = deep_sdf.data.get_instance_filenames(data_source, train_split)

    print(len(instance_filenames), " vs ", len(latent_vectors))

    for i, latent_vector in enumerate(latent_vectors):
        print(os.path.normpath(instance_filenames[i]))
        if sys.platform.startswith('linux'):
            dataset_name, class_name, instance_name = os.path.normpath(instance_filenames[i]).split("/")
        else:
            dataset_name, class_name, instance_name = os.path.normpath(instance_filenames[i]).split("\\")
        instance_name = instance_name.split(".")[0]

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        mesh_filename = os.path.join(output_dir, instance_name)

        offset = None
        scale = None

        if not keep_normalized:

            normalization_params = np.load(
                ws.get_normalization_params_filename(
                    data_source, dataset_name, class_name, instance_name
                )
            )
            offset = normalization_params["offset"]
            scale = normalization_params["scale"]

        with torch.no_grad():
            warped = create_motion_field(
                decoder,
                latent_vector,
                mesh_filename,
                N=resolution,
                max_batch=int(2 ** 17),
                offset=offset,
                scale=scale
            )
            motion_data = sitk.GetImageFromArray(warped)
            sitk.WriteImage(motion_data, mesh_filename + ".nii.gz")



if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to generate a mesh given a latent code."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--output_dir",
        "-o",
        dest="output_dir",
        required=True,
        help="Output filename",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--keep_normalization",
        dest="keep_normalized",
        default=False,
        action="store_true",
        help="If set, keep the meshes in the normalized scale.",
    )
    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        type=int,
        default=256,
        help="Marching cube resolution.",
    )

    sys.argv = [r"generate_training_meshes.py",
                "--experiment", r"examples/cardiac",
                "-o", r"D:\XiaohanYuan\from_git\DeepImplicitTemplates\examples\cardiac\TrainingMeshes\2000\cardiac\phases\motion_field"
                ]
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    code_to_mesh(
        args.experiment_directory,
        args.checkpoint,
        args.output_dir,
        args.keep_normalized,
        args.resolution)
