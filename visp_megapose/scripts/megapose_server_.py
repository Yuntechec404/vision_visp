#!/usr/bin/env python
# ROS 1
import rospy
from visp_megapose.srv import Init, InitResponse
from visp_megapose.srv import Track, TrackResponse
from visp_megapose.srv import Render, RenderResponse
import transforms3d
import os
import json
import megapose_server
import numpy as np
from pathlib import Path
import torch
import pandas as pd
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model

megapose_models = {
    'RGB': ('megapose-1.0-RGB', False),
    'RGBD': ('megapose-1.0-RGBD', True),
    'RGB-multi-hypothesis': ('megapose-1.0-RGB-multi-hypothesis', False),
    'RGBD-multi-hypothesis': ('megapose-1.0-RGB-multi-hypothesis-icp', True),
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
                assert not mesh_path, f"there are multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldn't find the mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


class MegaPoseServer:
    def __init__(self):
        rospy.init_node('MegaPoseServer')
        rospy.loginfo('Starting MegaPose Server...')

        # Parameters
        mesh_dir = rospy.get_param('~mesh_dir', 'visp_megapose/data/models')
        mesh_dir = Path(mesh_dir).absolute()
        assert mesh_dir.exists(), 'Mesh directory does not exist, cannot start server'
        model_name = rospy.get_param('~megapose_models', 'RGB')
        model_name = megapose_models[model_name][0]
        num_workers = rospy.get_param('~num_workers', 4)
        optimize = rospy.get_param('~optimize', False)

        self.num_workers = num_workers
        self.object_dataset = make_object_dataset(mesh_dir)
        model_tuple = self._load_model(model_name)
        self.model_info = model_tuple[0]
        self.model = model_tuple[1]
        self.model.eval()
        self.model.bsz_images = 256
        self.camera_data = self._make_camera_data(camera_data)
        self.renderer = Panda3dSceneRenderer(self.object_dataset)

        # Services
        rospy.Service('initial_pose', Init, self.init_pose_callback)
        rospy.Service('track_pose', Track, self.track_pose_callback)
        rospy.Service('render_object', Render, self.render_object_callback)

        rospy.loginfo('MegaPose Server is ready!')

    def _load_model(self, model_name):
        return NAMED_MODELS[model_name], load_named_model(model_name, self.object_dataset, n_workers=self.num_workers).cuda()

    def _make_camera_data(self, camera_data):
        c = CameraData()
        c.K = camera_data['K']
        c.resolution = (camera_data['h'], camera_data['w'])
        c.z_near = 0.001
        c.z_far = 100000
        return c

    def init_pose_callback(self, req):
        # Implement the Init service callback
        response = InitResponse()
        # Handle the request and fill in the response
        return response

    def track_pose_callback(self, req):
        # Implement the Track service callback
        response = TrackResponse()
        # Handle the request and fill in the response
        return response

    def render_object_callback(self, req):
        # Implement the Render service callback
        response = RenderResponse()
        # Handle the request and fill in the response
        return response


if __name__ == '__main__':
    try:
        server = MegaPoseServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
