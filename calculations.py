import tensorflow as tf
import pickle
import cv2
import numpy as np
import posenet
from pose import Pose
from score import Score
from dtaidistance import dtw


class get_Score(object):
    def __init__(self, lookup='lookup.pickle'):
        self.pose = Pose()
        self.score = Score()
        self.saved_pose = pickle.load(open(lookup, 'rb'))
        self.weights = self.saved_pose["weights"]
        self.new_video_coordinates = []

    def get_action_coords_from_dict(self, action):
        for k, v in self.saved_pose.items():
            if k == action:
                (model_array, no_of_frames) = (v, v.shape[0])
        return model_array, no_of_frames

    def calculate_Score(self, video, action):
        with tf.compat.v1.Session() as sess:
            model_cfg, model_outputs = posenet.load_model(101, sess)
            reference_coordinates, reference_video_frames = self.get_action_coords_from_dict(action)
            cap = cv2.VideoCapture(video)
            new_video_frames = 0
            if cap.isOpened() is False:
                print("error in opening video")
            while cap.isOpened():
                ret_val, image = cap.read()
                if ret_val:
                    input_points = self.pose.getpoints(image, sess, model_cfg, model_outputs)
                    if len(input_points) == 0:
                        continue
                    input_new_coords = np.asarray(self.pose.roi(input_points)[0:34]).reshape(17, 2)
                    self.new_video_coordinates.append(input_new_coords)
                    new_video_frames = new_video_frames + 1
                else:
                    break
            cap.release()
            final_score, score_list = self.score.compare_34dim(np.asarray(self.new_video_coordinates),
                                                         np.asarray(reference_coordinates),
                                                         new_video_frames, reference_video_frames, self.weights)
        return final_score, score_list
