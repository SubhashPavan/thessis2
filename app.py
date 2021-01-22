import os
import sys
import cv2
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
#from imutils import face_utils
import argparse
import json
#import imutils
import time
import numpy as np
from PIL import Image
from io import BytesIO
#import dlib


model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        
                        ])
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

def detect_faces(img):
    
    rects, landmarks = face_detect.detect_face(np.array(img),80);#min face size is set to 80x80
    aligns = []
    positions = []
    face_boundries = []
    for (i, rect) in enumerate(rects):
        aligned_face, face_pos = aligner.align(160,np.array(img),landmarks[:,i])
        if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
            aligns.append(aligned_face)
            positions.append(face_pos)
        else: 
            print("Align face failed") #log        
    if(len(aligns) > 0):
        features_arr = extract_feature.get_features(aligns)
        face_boundaries = rects
    
    else:
        face_boundaries = None
    
    
    return face_boundaries




def main():
    print("[INFO] camera sensor warming up...")
    vs = cv2.VideoCapture(0); #get input from webcam
    detect_time = time.time()
    while True:
        _,frame = vs.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detect_faces(frame)
        if rects is not None:
            for (i,rect) in enumerate(rects):
                #shape = predictor(gray,rect)
                #shape = face_utils.shape_to_np(shape)
                cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0))
                #for (x, y) in shape:
                #    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1
                #draw bounding box for the face
                
        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break     
        
        
    

if __name__ == "__main__":
    FRGraph = FaceRecGraph()
    MTCNNGraph = FaceRecGraph()
    aligner = AlignCustom()
    extract_feature = FaceFeature(FRGraph)
    face_detect = MTCNNDetect(MTCNNGraph, scale_factor=2); #scale_factor, rescales image for faster detection
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
    main()
