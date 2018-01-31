"""Lightwight version of the people detector app.

In order to run a lite object detection model as SSD Mobilenet through HTTP 
it's necessary to have access to a considerable amount of computing power, such
as a high queality dedicated graphic card. If that's not the case, this script 
limits the time taken for the algorithm to be feed with an image in order to 
make it more responsive since there's no control about the FPS of the source

"""

import os
import cv2
import numpy as np
import tensorflow as tf
import threading

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

try: 
    IMAGE_REFRESH = float(os.environ['IMAGE_REFRESH']) 
    VIDEO_SOURCE = os.environ['VIDEO_IP']
    DOCKER_PORT = int(os.environ['DOCKER_PORT'])
except KeyError: 
    IMAGE_REFRESH = 0.5
    VIDEO_SOURCE = 'http://10.68.8.119:8080/video.ogg'
    DOCKER_PORT = 8082

# Turn down all debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

CWD_PATH = os.getcwd()
TF_MODELS_PATH = CWD_PATH
# Path to frozen detection graph. This is the actual model that is used for the object detection
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

PATH_TO_CKPT = os.path.join(TF_MODELS_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

class TotalPeople:
    """Static variables accesible for the main and the rest of the threads.
    
    Args:
    i (int): Total number of people detected.
    img (nparray): Encoded image with the results of the prediction.
    data_dict (dict): Parsing destination of the JSON POST from /temperature.
        temperature (float): Temperature of the room.
    frame (ndarray): Frame read in a certain moment.
    
    """
    i = 0
    img = 0
    data_dict = {'temperature': None}
    frame = 0

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def model_load_into_memory():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    # Actual detection
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    
    # Transforming classes and scores into np
    classes_np = np.squeeze(classes).astype(np.int32)
    scores_np = np.squeeze(scores)
    
    # Draw just class 1 detections (Person)
    total_people = 0        # Total observations per frame
    TotalPeople.i = total_people
    for i in range(classes_np.size):
        if classes_np[i]==1 and scores_np[i]>=0.5:
            total_people += 1
            TotalPeople.i = total_people
        elif classes_np[i] != 1:
            scores_np[i] = 0.02
                
    # Visualization of the results of a detection
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        scores_np,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=.5, 
        line_thickness=8)
    return image_np

from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# Serving a web interface
class HTTPhandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
    def _set_image_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'image/jpeg')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_GET(self):
        """Looks in the target URL for the 'img' keyword.
        If so, loads into it the processed image.
        If not, loads the total number of people detected.
        
        """
        if 'img' in self.path:
            self._set_image_headers()
            content = TotalPeople.img
            self.wfile.write(content[1].tobytes())
        elif 'total' in self.path:
            self._set_headers()
            message = str(TotalPeople.i)
            self.wfile.write(bytes(message, "utf8"))
        elif 'temperature' in self.path:
            self._set_headers()
            message = str(TotalPeople.data_dict['temperature'])
            self.wfile.write(bytes(message, "utf8"))
        else:
            # Loading the resulting web and serving it again
            self._set_headers()
            html_file = open("ui.html", 'r', encoding='utf-8')
            source_code = html_file.read()
            self.wfile.write(bytes(source_code, "utf8"))
            
    def do_POST(self):
        if 'temperature' in self.path:
            content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
            post_data = self.rfile.read(content_length) # <--- Gets the data itself
            TotalPeople.data_dict = json.loads(post_data)
            self._set_headers()
      
    # Overriding log messages    
    def log_message(self, format, *args):
        return
        
class HTTPThread(threading.Thread):
    def __init__(self, name):
        super(HTTPThread, self).__init__()
        self.name = name
        self._stop_event = threading.Event()
        
    def run(self):
        server_address = ('', DOCKER_PORT)
        self.httpd = HTTPServer(server_address, HTTPhandler)
        self.httpd.serve_forever()

    def stop(self):
       self.httpd.shutdown()
       self.stopped = True
       
    def stopped(self):
       return self._stop_event.is_set()
   
from threading import Timer
   
class DetectionThread(threading.Thread):
    def __init__(self, detection_graph, sess, cap):
        super(DetectionThread, self).__init__()
        self.detection_graph = detection_graph
        self.sess = sess
        self.cap = cap
        self._stop_event = threading.Event()
        
    def single_detection(self):
        output = detect_objects(TotalPeople.frame, self.sess, self.detection_graph)
        TotalPeople.img = cv2.imencode('.jpeg', output)
        Timer(IMAGE_REFRESH, self.single_detection).start()

    def stop(self):
       self.stopped = True
       
    def stopped(self):
       return self._stop_event.is_set()

def main():
    detection_graph = model_load_into_memory()
    print("Algoritmo de reconocimiento en funcionamiento. Presionar \'Ctrl + C\' para detener la ejecución")
    
    # Thread starting in background
    http_thread = HTTPThread("HTTP Publisher Thread")
    http_thread.daemon = True
    http_thread.start()
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            detection_thread = DetectionThread(detection_graph, sess, cap)
            detection_thread.daemon = True
            Timer(IMAGE_REFRESH, detection_thread.single_detection).start()
            
            try:
                while True:
                    ret, TotalPeople.frame = cap.read()

            except KeyboardInterrupt:
                pass
                                  
            # Ending resources
            detection_thread.stop()
            http_thread.stop()
                        
if __name__ == "__main__":
    main()
    