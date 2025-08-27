import os
import sys
import json
import shutil
import argparse
import subprocess
import pathlib as Pathlib

import numpy as np

import source.util.util as util
import source.datasets_analyzer.scannet_v1.sensor_data as sensor_data

def generate_keyframe(opt, dataset_dir, images_stride=1, images_rate=1.0):
    json_data = dict()
    
    print("generating keyframe metadata...")

    length = len(os.listdir(os.path.join(dataset_dir, opt.filename, "color")))
    json_data[opt.filename] = [frame_count for frame_count in range(0, int(length * images_rate))]
    keyframe_dir = os.path.join(dataset_dir, "..").replace("\\", "/")
    with open(os.path.join(keyframe_dir, "keyframes.json"), 'w') as f:
        json.dump(json_data, f)

    print(f"keyframe metadata saved to {keyframe_dir}/keyframes.json")
    print("keyframe metadata generation complete.")

def generate_pose(opt, dataset_dir):
    print("generating pose metadata...")

    pose_dir = os.path.join(dataset_dir, opt.filename, "pose").replace("\\", "/")
    poses = [np.loadtxt(os.path.join(pose_dir, pose_name)) for pose_name in os.listdir(pose_dir)]
    output_pose_path = os.path.abspath(os.path.join(pose_dir, "..", "pose.npy"))
    np.save(output_pose_path, poses)
    shutil.rmtree(pose_dir)

    print(f"pose metadata saved to {pose_dir}/../pose.npy")
    print("pose metadata generation complete.")
    
config = util.load_config("./config/config.yml")

parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--filename', required=True, help='path to sens file to read')
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_poses', dest='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)

opt = parser.parse_args()
print(opt)

def main():
    output_path = os.path.join(config.dataset_dir, opt.filename).replace("\\", "/")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images_stride = config.images_stride if config.images_stride != None else 1 
    images_rate = config.images_rate if config.images_rate != None else 1.0

    # load the data
    data_dir = os.path.join(config.dataset_dir, '..', 'scans', opt.filename, opt.filename + ".sens").replace("\\", "/")
    sys.stdout.write('loading "%s"...\n' % data_dir)
    sd = sensor_data.SensorData(data_dir)
    if opt.export_depth_images:
        sd.export_depth_images(os.path.join(output_path, 'depth'), images_stride)
    if opt.export_color_images:
        sd.export_color_images(os.path.join(output_path, 'color'), images_stride)
    if opt.export_poses:
        sd.export_poses(os.path.join(output_path, 'pose'), images_stride)
    if opt.export_intrinsics:
        sd.export_intrinsics(os.path.join(output_path, 'intrinsic'), images_stride)

    generate_keyframe(opt, config.dataset_dir)
    generate_pose(opt, config.dataset_dir)

    intrinsic_dir = os.path.join(config.dataset_dir, opt.filename, "intrinsic")
    for file in os.listdir(intrinsic_dir):
        shutil.move(os.path.join(intrinsic_dir, file), os.path.join(intrinsic_dir, ".."))
    shutil.rmtree(intrinsic_dir)

    if config.tsdf_dir != None and config.tsdf_dir != "":
        if not os.path.exists(config.tsdf_dir):
            os.makedirs(config.tsdf_dir)
        command = [
            "python",
            "source/reconstruction/generate_gt_tsdf.py",
            "--dataset-dir", config.dataset_dir,
            "--output-dir", config.tsdf_dir
        ]
        subprocess.run(command, check=True)
        
if __name__ == "__main__":
    main()