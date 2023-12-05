
from .extractor import Extractor
from .encoder import Encoder
from deepface.commons import functions, distance as dst
from .utils import euclidian_l2
import os
from PIL import Image
import numpy as np
import math
from viam.logging import getLogger

LOGGER = getLogger(__name__)

class Identifier:
    def __init__(self,
                 detector_backend:str,
                 extraction_threshold:float, 
                 grayscale:bool, 
                 enforce_detection:bool, 
                 align:bool,
                 model_name:str,
                 normalization:str,
                 dataset_path: str, 
                 labels_directories:dict, 
                 distance_metric_name:str, 
                 identification_threshold:float):
        
        self.model_name = model_name
        target_size = functions.find_target_size(model_name=model_name)
        
        self.extractor = Extractor(target_size=target_size, 
                                   detector_backend=detector_backend,
                                   extraction_threshold = extraction_threshold, 
                                   grayscale=grayscale,
                                   enforce_detection=enforce_detection, 
                                   align=align)
        
        self.encoder = Encoder(model_name = model_name,
                               align=align,
                               normalization=normalization)
        
        self.dataset_path = dataset_path
        self.labels_directories = labels_directories
        self.model_name = model_name
        self.known_embeddings = dict()
        if distance_metric_name == "cosine":
            self.distance = dst.findCosineDistance
        if distance_metric_name == "euclidean":
            self.distance = dst.findEuclideanDistance
        elif distance_metric_name == "euclidean_l2":
            self.distance = euclidian_l2
        
        if identification_threshold is None:
            self.identification_threshold = dst.findThreshold(model_name, distance_metric_name)
        else:
            self.identification_threshold = identification_threshold
        
    def compute_known_embeddings(self):
        for label in self.labels_directories:
            label_path = self.dataset_path+"/"+self.labels_directories[label]
            embeddings = []
            for file in os.listdir(label_path):
                if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                ):
                    im = Image.open(label_path+"/"+file).convert('RGB') #convert in RGB because png openened in RGBA
                    img = np.array(im)[...,::-1]
                    faces = self.extractor.extract_faces(img)
                    for face, _, _ in faces: 
                        embed = self.encoder.encode(face)
                        embeddings.append(embed)
            self.known_embeddings[label] = embeddings
            
    def get_detections(self, img):
        """
        compute face detections and identifcations in the input
        image.  Faces whose minimum distance at best embedding are
        greater than the threshold are labelled as 'unknown'.
        Args:
            img (np.array): BGR format

        Returns:
            [Detections]: _description_
        """        
        detections = []
        faces = self.extractor.extract_faces(img)
        LOGGER.error(f"len(faces) is {len(faces)}")
        for face, face_region, _ in faces:
            match, dist = self.compare_face_to_known_faces(face)
            if match is not None:
                detection =  { "confidence": dist/self.identification_threshold,
                                 "class_name": match,
                                 "x_min": face_region["x"],
                                 "y_min": face_region["y"], 
                                 "x_max": face_region["x"] + face_region["w"],
                                "y_max": face_region["y"] + face_region["h"]}
            else:
                detection =  { "confidence": 1-dist/self.identification_threshold,
                                 "class_name": "unknown",
                                 "x_min": face_region["x"],
                                 "y_min": face_region["y"], 
                                 "x_max": face_region["x"] + face_region["w"],
                                "y_max": face_region["y"] + face_region["h"]}
            
            detections.append(detection)    
        
        return detections
            
    def compare_face_to_known_faces(self, face):
        """
        Encodes the face, calculates its distances with known faces and
        returns the best match. If no match is found, returns (None, math.inf).

        Args:
            face (np.array): extracted face of size self.target_size

        Returns:
            (label, distance):
        """
        source_embed = self.encoder.encode(face)
        match = None
        min_dist = math.inf
        for label in self.known_embeddings:
            for target_embed in self.known_embeddings[label]:
                dist = self.distance(source_embed, target_embed)
                if dist <=self.identification_threshold and dist<min_dist:
                    match, min_dist  = label, dist
        return match, min_dist