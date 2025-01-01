#!/home/user/anaconda3/envs/megapose/bin/python3
# -*- coding: utf-8 -*-
import sys
import rospy
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
# import megapose_server


rospy.init_node("test_node")
rospy.loginfo(sys.executable)
rospy.loginfo(sys.path)
