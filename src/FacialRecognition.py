from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from imutils.video import VideoStream
from typing import List, Tuple
import argparse
import facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
from pyzbar.pyzbar import decode


class FaceDetector:
    def __init__(self, minsize: int, threshold: List[float], factor: float, gpu_memory_fraction: float, detect_face_model_path: str, facenet_model_path: str):
        self.minsize = minsize
        self.INPUT_IMAGE_SIZE=160
        self.threshold = threshold
        self.factor = factor
        facenet.load_model(facenet_model_path)
        self.gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False))
        self.pnet, self.rnet, self.onet = self.create_mtcnn(detect_face_model_path)
        self.images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

    def create_mtcnn(self, detect_face_model_path: str) -> Tuple:
        with tf.Graph().as_default():
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, detect_face_model_path)

    def get_faces(self, frame: np.ndarray) -> List:
        return align.detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)

    def get_embeddings(self, face: np.ndarray) -> np.ndarray:
        scaled = cv2.resize(face, (self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        scaled = facenet.prewhiten(scaled)
        scaled_reshape = scaled.reshape(-1, self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, 3)
        feed_dict = {self.images_placeholder: scaled_reshape, self.phase_train_placeholder: False}
        return self.sess.run(self.embeddings, feed_dict=feed_dict)


class FaceRecognition:
    def __init__(self, classifier_path: str):
        with open(classifier_path, 'rb') as file:
            self.model, self.class_names = pickle.load(file)
        print("Custom Classifier, Successfully loaded")

    def recognize_face(self, embeddings: np.ndarray, threshold: float = 0.8) -> Tuple[str, float]:
        predictions = self.model.predict_proba(embeddings)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        best_name = self.class_names[best_class_indices[0]] if best_class_probabilities[0] > threshold else "unknown"
        return best_name, float(best_class_probabilities[0])
class BarcodeReader:
    # def __init__(self, verbose: bool = False):
    #     self.verbose = verbose
    # def read_barcodes(self, frame: np.ndarray) -> List[str]:
    #     decoded_objs = decode(frame)
    #     barcodes = []
    #     for obj in decoded_objs:
    #         barcodes.append(obj.data.decode('utf-8'))
    #         if self.verbose:
    #             print("Barcode: ", barcodes[-1])
    #     return barcodes
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    def read_barcodes(self, frame: np.ndarray) -> List[str]:
        # draw bounding box around barcode and display barcode type and data to terminal
        decoded_objs = decode(frame)
        barcodes = []
        for obj in decoded_objs:
            barcodes.append(obj.data.decode('utf-8'))
            # Draw bounding box around barcode
            (x, y, w, h) = obj.rect
            #viết chữ lên ảnh
            cv2.putText(frame, barcodes[-1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if self.verbose:
                print("Barcode: ", barcodes[-1])
        return barcodes
def main(args):
    # Load face detector and recognizer
    detector = FaceDetector(
    minsize=args["minsize"],
    threshold=args["threshold"],
    factor=args["factor"],
    gpu_memory_fraction=args["gpu_memory_fraction"],
    detect_face_model_path=args["detect_face_model_path"],
    facenet_model_path=args["facenet_model_path"]
    )
    recognizer = FaceRecognition(classifier_path=args["classifier_path"])
    # Initialize barcode reader
    barcode_reader = BarcodeReader(verbose=args["verbose"])

    # Start video stream
    print("[INFO] Starting video stream...")
    vs = VideoStream(src=args["video_path"]).start()

    # Loop over frames from the video stream
    while True:
        # Read the next frame from the video stream
        frame = vs.read()

        # If the frame is None, then we have reached the end of the stream
        if frame is None:
            break

        # Convert the frame from BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        faces, _= detector.get_faces(rgb)

        # Loop over detected faces
        for face in faces:
            try:
                print (face[:4])
                # Get face coordinates
                x1, y1, x2, y2 = face[:4]

                # Get face image
                face_img = rgb[int(y1):int(y2), int(x1):int(x2), :]

                # Get face embeddings
                embeddings = detector.get_embeddings(face_img)

                # Recognize face
                name, prob = recognizer.recognize_face(embeddings)

                # Draw face bounding box and name
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, "{} {:.2f}".format(name, prob), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except:
                pass

        # Read barcodes from the frame
        barcodes = barcode_reader.read_barcodes(frame)

        # Show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # If the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # Stop the video stream and close all windows
    vs.stop()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--minsize', type=int, default=20, help='Minimum size of face to detect')
    parser.add_argument('--threshold', type=list, default=[0.6, 0.7, 0.7], help='Threshold for the MTCNN model')
    parser.add_argument('--factor', type=float, default=0.709, help='Scale factor for the MTCNN model')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.6, help='GPU memory fraction to allocate')
    parser.add_argument('--detect_face_model_path', type=str, default='./src/align',
    help='Path to the MTCNN face detection model')
    parser.add_argument('--facenet_model_path', type=str, default='./Models/20180402-114759.pb',
    help='Path to the facenet model')
    parser.add_argument('--classifier_path', type=str, default='./Models/facemodel.pkl',
    help='Path to the classifier model')
    parser.add_argument('--video_path', type=str, default=0, help='Path to the input video')
    parser.add_argument('--verbose', type=bool, default=True, help='Enable verbose mode')
    main(vars(parser.parse_args()))
