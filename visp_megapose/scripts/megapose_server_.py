#!/home/user/anaconda3/envs/megapose/bin/python3
# -*- coding: utf-8 -*-
import rospy
from visp_megapose.srv import Init, Track, Render
# import transforms3d

import os
import json
import megapose_server
from sensor_msgs.msg import Image
from std_msgs.msg import Header

# Import necessary libraries
import numpy as np
from pathlib import Path
import torch
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.inference.types import ObservationTensor
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose_server.server_operations import ServerMessage

megapose_server_install_dir = os.path.dirname(megapose_server.__file__)
variables_file = os.path.join(megapose_server_install_dir, 'megapose_variables_final.json')
with open(variables_file, 'r') as f:
    json_vars = json.load(f)
    rospy.loginfo(f"Loaded megapose variables {json_vars}")
    os.environ['MEGAPOSE_DIR'] = json_vars['megapose_dir']
    os.environ['MEGAPOSE_DATA_DIR'] = json_vars['megapose_data_dir']

if 'HOME' not in os.environ:  # Home is always required by megapose but is not always set
    os.environ['HOME'] = os.getenv('HOMEPATH', os.getenv('HOMEDIR', '.'))



megapose_models = {
    'RGB': ('megapose-1.0-RGB', False),
    'RGBD': ('megapose-1.0-RGBD', True),
}

camera_data = {
    'K': np.asarray([
        [700, 0.0, 320],
        [0.0, 700, 240],
        [0.0, 0.0, 1.0]
    ]),
    'h': 480,
    'w': 640
}


def make_object_dataset(meshes_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "m"
    object_dirs = meshes_dir.iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply", ".glb", ".gltf"}:
                assert not mesh_path, f"Multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"Couldn't find the mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    return RigidObjectDataset(rigid_objects)


class MegaPoseServer:
    def __init__(self):
        rospy.init_node('MegaPoseServer', anonymous=True)
        self.mesh_dir = rospy.get_param('~mesh_dir', 'visp_megapose/data/models')
        self.model_name = rospy.get_param('~megapose_models', 'RGB')
        self.num_workers = rospy.get_param('~num_workers', 4)

        self.mesh_dir = Path(self.mesh_dir).absolute()
        assert self.mesh_dir.exists(), 'Mesh directory does not exist'
        self.object_dataset = make_object_dataset(self.mesh_dir)

        model_name = megapose_models[self.model_name][0]
        self.model_info, self.model = self._load_model(model_name)
        self.model.eval()
        self.camera_data = self._make_camera_data(camera_data)
        self.renderer = Panda3dSceneRenderer(self.object_dataset)

        # ROS Services
        self.srv_initial_pose = rospy.Service('initial_pose', Init, self.InitPoseCallback)
        self.srv_track_pose = rospy.Service('track_pose', Track, self.TrackPoseCallback)
        self.srv_render_object = rospy.Service('render_object', Render, self.RenderObjectCallback)
        rospy.loginfo("MegaPoseServer initialized and waiting for requests...")

    def _load_model(self, model_name):
        return NAMED_MODELS[model_name], load_named_model(model_name, self.object_dataset, n_workers=self.num_workers)

    def _make_camera_data(self, camera_data):
        c = camera_data.copy()
        c['z_near'] = 0.001
        c['z_far'] = 100000
        return c

    def InitPoseCallback(self, request):
        # Add logic to process the request and provide a response
        pass

    def TrackPoseCallback(self, request):
        # Add logic to process the request and provide a response
        pass

    def RenderObjectCallback(self, request):
        # Add logic to process the request and provide a response
        pass


if __name__ == '__main__':
    server = MegaPoseServer()
    rospy.spin()
