#!/usr/bin/env python3
""" Yolo class """

import tensorflow as tf
import tensorflow.keras as K
import numpy as np


class Yolo:
    """ Class Yolo v3 """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ constructor for a Yolo class instance

        Agruments:
            - model_path: path to where a Darknet Keras model is stored
            - classes_path: path to where the list of class names used for the
                    Darknet model, listed in order of index, can be found
            - class_t: float representing the box score threshold for the
                    initial filtering step
            - nms_t: float representing the IOU threshold for
                    non-max suppression
            - anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                    containing all of the anchor boxes:
                - outputs: number of outputs (predictions)
                        made by the Darknet model
                - anchor_boxes: number of anchor boxes used for each prediction
                - 2: [anchor_box_width, anchor_box_height]

        Returns:
            None
        """
        self.model = K.models.load_model(model_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

        with open(classes_path, "r") as classes_file:
            self.class_names = []
            for class_name in classes_file:
                self.class_names.append(class_name.strip())

    def sigmoid(self, x):
        """ Sigmoïd function """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """ process outputs

        Arguments:
            - outputs: list of numpy.ndarrays containing the predictions
                    from the Darknet model for a single image:
                - Each output will have the shape (grid_height, grid_width,
                        anchor_boxes, 4 + 1 + classes)
                    - grid_height: height of the grid used for the output
                    - grid_width: width of the grid used for the output
                    - anchor_boxes: number of anchor boxes used
                    - 4: (t_x, t_y, t_w, t_h)
                    - 1: box_confidence
                    - classes: class probabilities for all classes
            - image_size: numpy.ndarray containing the image's original size
                    [image_height, image_width]

        Returns:
            - boxes: list of numpy.ndarrays of shape (grid_height, grid_width,
                    anchor_boxes, 4) containing the processed boundary boxes
                    for each output, respectively:
                - 4: (x1, y1, x2, y2)
                - (x1, y1, x2, y2): should represent the boundary box relative
                    to original image
            - box_confidences: list of numpy.ndarrays of shape (grid_height,
                    grid_width, anchor_boxes, 1) containing the box confidences
                    for each output, respectively
        """
        boxes, box_confidences, box_class_probs = [], [], []
        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height = output.shape[0]
            grid_width = output.shape[1]
            anchor_boxes = output.shape[2]

            boxes.append(output[:, :, :, 0:4])
            box_confidences.append(self.sigmoid(output[:, :, :, 4:5]))
            box_class_probs.append(self.sigmoid(output[:, :, :, 5:]))

            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]
            c_x = np.indices((grid_height, grid_width, anchor_boxes))[1]
            c_y = np.indices((grid_height, grid_width, anchor_boxes))[0]
            b_x = self.sigmoid(t_x) + c_x
            b_y = self.sigmoid(t_y) + c_y
            # normalization to have b_x and b_y by grid
            b_x /= grid_width
            b_y /= grid_height

            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]
            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]
            b_w = p_w * np.exp(t_w)
            b_h = p_h * np.exp(t_h)
            # normalization by input of darknet model
            b_w /= input_width
            b_h /= input_height

            x1 = (b_x - (b_w / 2)) * image_width
            y1 = (b_y - (b_h / 2)) * image_height
            x2 = (b_x + (b_w / 2)) * image_width
            y2 = (b_y + (b_h / 2)) * image_height

            boxes[i][:, :, :, 0] = x1
            boxes[i][:, :, :, 1] = y1
            boxes[i][:, :, :, 2] = x2
            boxes[i][:, :, :, 3] = y2

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ filter boxes

        Arguments:
            - boxes: list of numpy.ndarrays of shape (grid_height, grid_width,
                    anchor_boxes, 4) containing the processed boundary boxes
                    for each output, respectively:
                - 4: (x1, y1, x2, y2)
            - box_confidences: list of numpy.ndarrays of shape (grid_height,
                    grid_width, anchor_boxes, 1) containing the processed box
                    confidences for each output, respectively
            - box_class_probs: list of numpy.ndarrays of shape (grid_height,
                    grid_width, anchor_boxes, classes) containing the processed
                    box class probabilities for each output, respectively

        Returns:
            - boxes: a tuple of (filtered_boxes, box_classes, box_scores):
                - filtered_boxes: a numpy.ndarray of shape (?, 4) containing
                        all of the filtered bounding boxes:
                - box_classes: a numpy.ndarray of shape (?,) containing the
                        class number that each box in filtered_boxes predicts,
                        respectively
                - box_scores: a numpy.ndarray of shape (?) containing the box
                        scores for each box in filtered_boxes, respectively
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        grid_height = box_class_probs[0].shape[0]
        grid_width = box_class_probs[0].shape[1]
        anchor_boxes = box_class_probs[0].shape[2]
        classes = box_class_probs[0].shape[3]

        for i in range(len(boxes)):
            for h in range(grid_height):
                for w in range(grid_width):
                    for anchor in range(anchor_boxes):
                        # get the box_score = box_confidence * class_prob
                        box_conf = box_confidences[i][h][w][anchor][0]
                        scores = []
                        for num_classe in range(classes):
                            class_prob = box_class_probs[
                                i][h][w][anchor][num_classe]
                            box_score = box_conf * class_prob
                            scores.append(box_score)

                        if max(scores) >= self.class_t:
                            filtered_boxes.append(boxes[i][h][w][anchor])
                            index_max = scores.index(max(scores))
                            box_classes.append(index_max)
                            class_prob = box_class_probs[
                                i][h][w][anchor][index_max]
                            box_score = box_conf * class_prob
                            box_scores.append(box_score)

            np_filtered_boxes = np.asarray(filtered_boxes)
            np_box_classes = np.asarray(box_classes)
            np_box_scores = np.asarray(box_scores)

            return np_filtered_boxes, np_box_classes, np_box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
           Args:
            filtered_boxes: a numpy.ndarray of shape (?, 4)
              containing all of the filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,) containing the class
              number for the class that filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing the box scores
              for each box in filtered_boxes, respectively
           Return:
            Tuple (box_predictions, predicted_box_class, predicted_box_scores)
            box_predictions: a numpy.ndarray of shape (?, 4) containing all of
              the predicted bounding boxes ordered by class and box score
            predicted_box_classes: a numpy.ndarray of shape (?,) containing
              the class number for box_predictions ordered by
              class and box score, respectively
            predicted_box_scores: a numpy.ndarray of shape (?) containing
              the box scores for box_predictions ordered by class and
              box score, respectively
        """
        # Non max suppression
        idx = tf.image.non_max_suppression(
            filtered_boxes, box_scores, box_scores.shape[0],
            iou_threshold=self.nms_t
        )
        run = K.backend.eval
        sup_boxes = run(tf.gather(filtered_boxes, idx))
        sup_scores = run(tf.gather(box_scores, idx))
        sup_classes = run(tf.gather(box_classes, idx))

        # Sort by class
        srt = sup_classes.argsort()
        sup_classes = sup_classes[srt]
        sup_scores = sup_scores[srt]
        sup_boxes = sup_boxes[srt, :]

        # Get indexes for sorting by score within
        # within each each group pre sorted by class
        idxs = []
        for x in range(81):
            idx_chunk = np.where(sup_classes == x)
            if idx_chunk[0].shape[0] > 0:
                idxs.append(np.amax(idx_chunk))
        prev = 0

        for x in idxs:
            # ordered slice of box scores
            slice = (-sup_scores[prev:x+1]).argsort()
            sup_scores[prev:x+1] = (sup_scores[prev:x+1])[slice]
            sup_boxes[prev:x+1, :] = (sup_boxes[prev:x+1, :])[slice]
            prev = x+1

        return (sup_boxes, sup_classes, sup_scores)