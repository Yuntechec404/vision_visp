#!/usr/bin/env python
import rospy
import json
import numpy as np
from pathlib import Path
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.inference.types import ObservationTensor, PoseEstimatesType
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.lib3d.transform import Transform

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose
import torch

class MegaposeServerROS:
    def __init__(self):
        rospy.init_node('megapose_server', anonymous=True)

        # Parameters
        self.mesh_dir = Path(rospy.get_param("~mesh_dir", "./meshes"))
        self.model_name = rospy.get_param("~model_name", "megapose-1.0-RGB")
        self.num_workers = rospy.get_param("~num_workers", 4)
        self.image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/rgb/camera_info")
        self.detections_topic = rospy.get_param("~detections_topic", "/detections")
        self.pose_topic = rospy.get_param("~pose_topic", "/estimated_poses")

        # Load object dataset and model
        self.object_dataset = self.make_object_dataset(self.mesh_dir)
        self.model_info, self.model = self.load_model(self.model_name)
        self.model.eval()
        self.renderer = Panda3dSceneRenderer(self.object_dataset)

        # ROS Publishers and Subscribers
        self.pose_pub = rospy.Publisher(self.pose_topic, PoseArray, queue_size=10)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        self.detections_sub = rospy.Subscriber(self.detections_topic, String, self.detections_callback)

        self.camera_data = None
        rospy.loginfo("Megapose Server ROS Node Initialized")

    def make_object_dataset(self, meshes_dir):
        rigid_objects = []
        mesh_units = "m"
        object_dirs = meshes_dir.iterdir()
        for object_dir in object_dirs:
            label = object_dir.name
            mesh_path = next((fn for fn in object_dir.glob("*") if fn.suffix in {".obj", ".ply", ".glb", ".gltf"}), None)
            assert mesh_path, f"Could not find mesh for {label}"
            rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        return RigidObjectDataset(rigid_objects)

    def load_model(self, model_name):
        return NAMED_MODELS[model_name], load_named_model(model_name, self.object_dataset, n_workers=self.num_workers).cuda()

    def camera_info_callback(self, msg):
        self.camera_data = {
            'K': np.array(msg.K).reshape(3, 3),
            'resolution': (msg.height, msg.width)
        }
        rospy.loginfo("Camera info received and updated")

    def image_callback(self, msg):
        # Convert ROS Image message to numpy array
        try:
            from cv_bridge import CvBridge
            bridge = CvBridge()
            image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rospy.loginfo("Image received")
            self.process_image(image)
        except Exception as e:
            rospy.logerr(f"Failed to process image: {e}")

    def detections_callback(self, msg):
        # Handle detections (example placeholder)
        rospy.loginfo(f"Detections received: {msg.data}")

    def process_image(self, image):
        if self.camera_data is None:
            rospy.logwarn("Camera data not yet available")
            return

        observation = ObservationTensor.from_numpy(image, None, self.camera_data['K']).cuda()
        inference_params = self.model_info['inference_parameters']
        labels = list(self.object_dataset.label_to_objects.keys())
        detections = None  # Placeholder, add detection handling here
        output, _ = self.model.run_inference_pipeline(observation, detections=detections, **inference_params)

        self.publish_poses(output)

    def publish_poses(self, output):
        pose_array = PoseArray()
        for pose in output.poses.cpu().numpy():
            p = Pose()
            p.position.x, p.position.y, p.position.z = pose[:3, 3]
            # Add quaternion conversion here if needed
            pose_array.poses.append(p)
        self.pose_pub.publish(pose_array)
        rospy.loginfo("Published estimated poses")


if __name__ == '__main__':
    try:
        server = MegaposeServerROS()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Megapose Server ROS Node Shutting Down")
