import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
from PIL import Image
import dlib
try:
    predictor = dlib.shape_predictor(r"C:\Users\Dee Pen\project_directory")
    print("Shape predictor loaded successfully.")
except RuntimeError as e:
    print(f"Error loading shape predictor: {e}")
