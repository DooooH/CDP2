import os

import tensorflow as tf
import cv2
import time
import math
import numpy
import numpy as np
import posenet
from pose import Pose
from score import Score
import pickle
import argparse

# USAGE : python3 keypoints_from_video.py --activity "punch - side" --video "test.mp4"

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--activity", required=True,
                help="activity to be recorder")
ap.add_argument("-v", "--video", required=True,
                help="video file from which keypoints are to be extracted")
ap.add_argument("-l", "--lookup", default="lookup_new.pickle",
                help="The pickle file to dump the lookup table")
args = vars(ap.parse_args())


def main():
    pose = Pose()
    coordinate_list = []

    # importing the module
    import json

    # Opening JSON file
    with open('weights.json') as json_file:
        weights = json.load(json_file)["weights"]

        s = sum(weights)
        weights = [w / s for w in weights]

    with tf.compat.v1.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(101, sess)

        cap = cv2.VideoCapture(args["video"])
        i = 1

        if cap.isOpened() is False:
            print("error in opening video")
        while cap.isOpened():
            ret_val, image = cap.read()
            if ret_val:
                input_points, input_black_image = pose.getpoints_vis(image, sess, model_cfg, model_outputs)
                cv2.imwrite(
                    './test_video' + str(i) + '.jpg',
                    input_black_image)
                input_points = input_points[0:34]
                # print(input_points)
                input_new_coords = pose.roi(input_points)
                input_new_coords = input_new_coords[0:34]
                input_new_coords = np.asarray(input_new_coords).reshape(17, 2)
                coordinate_list.append(input_new_coords)
                i = i + 1
            else:
                break
        cap.release()

        coordinate_list = np.array(coordinate_list)

        # print(coordinate_list)
        # print(coordinate_list.shape)
        print("Lookup Table Created")
        file = open(args["lookup"], 'wb')
        pickle.dump({args["activity"]: coordinate_list, "weights": weights}, file)
    # pickle.dump()


if __name__ == "__main__":
    main()
