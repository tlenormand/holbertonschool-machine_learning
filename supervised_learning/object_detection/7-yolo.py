#!/usr/bin/env python3
""" Yolo class """

import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import glob
import cv2


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
                    - 1: boxes_confidence
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
        image_height, image_width = image_size[0], image_size[1]
        boxes = [output[..., :4] for output in outputs]
        boxes_confidence, classes_probs = [], []
        corner_x, corner_y = [], []

        for output in outputs:
            grid_height, grid_width, anchors = output.shape[:3]
            # cx for center of gravity of the grid along width
            cx = np.arange(grid_width).reshape(1, grid_width)
            cx = np.repeat(cx, grid_height, axis=0)
            # cy for center of gravity of the grid along height
            cy = np.arange(grid_width).reshape(1, grid_width)
            cy = np.repeat(cy, grid_height, axis=0).T

            corner_x.append(np.repeat(cx[..., np.newaxis], anchors, axis=2))
            corner_y.append(np.repeat(cy[..., np.newaxis], anchors, axis=2))
            # box confidence and class probability activations
            boxes_confidence.append(self.sigmoid(output[..., 4:5]))
            classes_probs.append(self.sigmoid(output[..., 5:]))

        input_width = self.model.input.shape[1].value
        input_height = self.model.input.shape[2].value

        # Predicted boundary box
        for x, box in enumerate(boxes):
            bx = ((self.sigmoid(box[..., 0]) + corner_x[x]) /
                  outputs[x].shape[1])
            by = ((self.sigmoid(box[..., 1]) + corner_y[x]) /
                  outputs[x].shape[0])
            bw = ((np.exp(box[..., 2]) * self.anchors[x, :, 0]) / input_width)
            bh = ((np.exp(box[..., 3]) * self.anchors[x, :, 1]) / input_height)

            # x1
            box[..., 0] = (bx - (bw * 0.5))*image_width
            # y1
            box[..., 1] = (by - (bh * 0.5))*image_height
            # x2
            box[..., 2] = (bx + (bw * 0.5))*image_width
            # y2
            box[..., 3] = (by + (bh * 0.5))*image_height

        return boxes, boxes_confidence, classes_probs

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
        best_boxes, scores, classes = [], [], []

        for x in range(len(boxes)):
            box_score = box_confidences[x] * box_class_probs[x]
            box_class = np.argmax(box_score, axis=-1)
            box_score = np.amax(box_score, axis=-1)
            mask = box_score >= self.class_t

            if best_boxes == []:
                best_boxes = boxes[x][mask]
                scores = box_score[mask]
                classes = box_class[mask]
            else:
                best_boxes = np.concatenate(
                    (best_boxes, boxes[x][mask]), axis=0
                )
                scores = np.concatenate((scores, box_score[mask]), axis=0)
                classes = np.concatenate((classes, box_class[mask]), axis=0)

        return (best_boxes, classes, scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ non max suppression

        Arguments:
            - filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
                    the filtered bounding boxes:
            - box_classes: a numpy.ndarray of shape (?,) containing the class
                    number for the class that filtered_boxes predicts
            - box_scores: a numpy.ndarray of shape (?) containing the box
                    scores for each box in filtered_boxes, respectively

        Returns:
            - boxes: a numpy.ndarray of shape (?, 4) containing all of the
                    filtered bounding boxes:
            - box_classes: a numpy.ndarray of shape (?,) containing the class
                    number for box in filtered_boxes, respectively
            - box_scores: a numpy.ndarray of shape (?) containing the box
                    scores for each box in filtered_boxes, respectively
        """
        # Non max suppression
        idx = tf.image.non_max_suppression(
            filtered_boxes, box_scores, box_scores.shape[0],
            iou_threshold=self.nms_t
        )
        sup_boxes = K.backend.eval(tf.gather(filtered_boxes, idx))
        sup_scores = K.backend.eval(tf.gather(box_scores, idx))
        sup_classes = K.backend.eval(tf.gather(box_classes, idx))

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
            prev = x + 1

        return (sup_boxes, sup_classes, sup_scores)

    @staticmethod
    def load_images(folder_path):
        """ load images

        Arguments:
            - folder_path: a string representing the path to the folder
                    holding all the images to load

        Returns:
            - images: a list of images as numpy.ndarrays
            - image_paths: a list of paths to the individual images in images
        """
        images = []
        image_paths = glob.glob(folder_path + '/*')

        for image in image_paths:
            images.append(cv2.imread(image))

        return images, image_paths

    def preprocess_images(self, images):
        """ preprocess images

        Arguments:
            - images: a list of images as numpy.ndarrays

        Returns:
            - pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
                    containing all of the preprocessed images
                    - ni: the number of images that were preprocessed
                    - input_h: the input height for the Darknet model
                            Note: this can vary by model
                    - input_w: the input width for the Darknet model
                            Note: this can vary by model
                    - 3: number of color channels
            - image_shapes: a numpy.ndarray of shape (ni, 2) containing the
                    original height and width of the images
        """
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]
        pimages = []

        for image in images:
            pimages.append(cv2.resize(
                image, (input_width, input_height),
                interpolation=cv2.INTER_CUBIC
            ) / 255)

        image_shapes = np.stack(
            [(image.shape[0], image.shape[1]) for image in images],
            axis=0
        )

        return np.array(pimages), image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """ show boxes

        Arguments:
            - image: a numpy.ndarray containing an unprocessed image
            - boxes: a numpy.ndarray containing the boundary boxes for the
                    image
            - box_classes: a numpy.ndarray containing the class indices for
                    each box
            - box_scores: a numpy.ndarray containing the box scores for each
                    box
            - file_name: the file path where the original image is stored

        Returns:
            - None
        """
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(
                image, (int(x1), int(y1)), (int(x2), int(y2)),
                (255, 0, 0), 2
            )
            text = f"{self.class_names[box_classes[i]]} {box_scores[i]:.2f}"
            cv2.putText(
                image, text, (int(x1), int(y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key in (ord('s'), ord('q')):
            cv2.destroyAllWindows()
            if key == ord('s'):
                cv2.imwrite(file_name, image)
            exit()

    def predict(self, folder_path):
        """ predict

        Arguments:
            - folder_path: a string representing the path to the folder
                    holding all the images to predict

        Returns:
            - predictions: a list of tuples for each image of
                    (boxes, box_classes, box_scores)
            - image_paths: a list of image paths corresponding to
                    each prediction in predictions
        """
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)
        outputs = self.model.predict(pimages)
        predictions = []

        for i in range(len(images)):
            image = images[i].copy()
            boxes, box_confidences, box_class_probs = self.process_outputs(
                outputs, image_shapes[i]
            )
            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confidences, box_class_probs
            )
            sup_boxes, sup_classes, sup_scores = self.non_max_suppression(
                filtered_boxes, box_classes, box_scores
            )
            self.show_boxes(
                image, sup_boxes, sup_classes, sup_scores, image_paths[i]
            )
            predictions.append((sup_boxes, sup_classes, sup_scores))

        return predictions, image_paths