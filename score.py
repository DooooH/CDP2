import numpy as np
from dtaidistance import dtw, dtw_ndim, ed


class Score(object):
    def percentage_score(self, score):  # To be replaced with a better scoring algorithm, if found in the future
        percentage = 100 - (score * 100)
        return int(percentage)

    def dtwdis(self, model_points, input_points, i, j):
        model_points = model_points.reshape(2 * j, )
        input_points = input_points.reshape(2 * i, )
        model_points = model_points / np.linalg.norm(model_points)
        input_points = input_points / np.linalg.norm(input_points)
        return self.percentage_score(dtw.distance(model_points, input_points))


    def dtwdis_new(self, model_points, input_points):
        return self.percentage_score(dtw_ndim.distance(model_points, input_points))


    def normalize(self, input_test):
        for k in range(0, 17):
            input_test[:, k] = input_test[:, k] / np.linalg.norm(input_test[:, k])
        return input_test


    def compare_separate(self, new_video_coordinates, reference_coordinates, i, j, weights):
        # new_video_coordinates = self.normalize(new_video_coordinates)
        scores = []
        for k in range(0, 17):
            # k 번째 관절
            scores.append(self.dtwdis(new_video_coordinates[:, k], reference_coordinates[:, k], i, j))
        return self.apply_weights(weights, scores), scores


    def compare_34dim(self, new_video_coordinates, reference_coordinates, i, j, weights):
        # new_video_coordinates = self.normalize(new_video_coordinates)
        scores = []
        new_video_coordinates = new_video_coordinates.reshape(i, 34)
        reference_coordinates = reference_coordinates.reshape(j, 34)
        best_path = dtw.best_path(dtw_ndim.warping_paths(new_video_coordinates, reference_coordinates)[1])
        # Calculating euclidean distance per body part to apply weights
        body_part_scores = []
        for body_part_i in range(17):
            v1_part, v2_part = [False * i * 2], [False * j * 2]
            for k, new_frame, reference_frame in enumerate(best_path):
                v1_part[k * 2] = new_video_coordinates[new_frame][body_part_i * 2]
                v1_part[k * 2 + 1] = new_video_coordinates[new_frame][body_part_i * 2 + 1]
                v2_part[k * 2] = reference_coordinates[reference_frame][body_part_i * 2]
                v2_part[k * 2 + 1] = reference_coordinates[reference_frame][body_part_i * 2 + 1]
            body_part_scores.append(ed.distance(v1_part, v2_part))

        return self.apply_weights(weights, body_part_scores), scores

    def apply_weights(self, weights, scores):
        return list(map(lambda z: z[0] * z[1], zip(np.array(weights), scores)))
