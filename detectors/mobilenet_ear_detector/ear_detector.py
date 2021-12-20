import cv2
import numpy as np
import os 
import ast

class Detector:

  # Load detected bboxes extracted with Google Colab
  def detect(self, name):
    boxes = []
    name = name.replace(".png", ".txt")
    name = name.replace("data/ears/test", "")
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    annot_name = os.path.join(THIS_FOLDER, "colab_bboxes_21300" + name)
    
    f = open(annot_name, "r")
    boxes = f.read()
    boxes = ast.literal_eval(boxes)
 
    return boxes

    """
    ### CODE WAS EXECTUED ON GOOGLE COLAB -- bounding boxes were extracted and imported into folders colab_bboxes_******
    with tf.gfile.FastGFile('data/detectors/mobilenet_ear_detector/frozen_inference_graph.pb', 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
      # Restore session
      sess.graph.as_default()
      tf.import_graph_def(graph_def, name='')

      # Read and preprocess an image.
      img = cv.imread(im_name)
      im_width = img.shape[0]
      im_height = img.shape[1]
      inp = cv.resize(img, (300, 300))
      inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

      # Run the model
      out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                      sess.graph.get_tensor_by_name('detection_scores:0'),
                      sess.graph.get_tensor_by_name('detection_boxes:0'),
                      sess.graph.get_tensor_by_name('detection_classes:0')],
                      feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

      # Visualize detected bounding boxes.
      num_detections = int(out[0][0])

      class_ids = []
      confidences = []
      boxes = []
      for i in range(num_detections):
          classId = int(out[3][0][i])
          score = float(out[1][0][i])
          bbox = [float(v) for v in out[2][0][i]]
          if score > 0.3:
            xmin, ymin, xmax, ymax = bbox
            (xmin, xmax, ymin, ymax) = (xmin * im_width, xmax * im_width,
                                            ymin * im_height, ymax * im_height)
            boxes.append([int(ymin), int(xmin), int(ymax-ymin), int(xmax-xmin)])
            confidences.append(float(score))
            class_ids.append(0)
    return boxes
    """
    #indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(indexes)
    #font = cv2.FONT_HERSHEY_PLAIN
    #for i in range(len(boxes)):
      #if i in indexes:
       # x, y, w, h = boxes[i]
        #label = str(classes[class_ids[i]])
        #color = colors[class_ids[i]]
        #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        #cv2.putText(img, label, (x, y + 30), font, 3, color, 2)


    #cv2.imshow("Image", img)
    #key = cv2.waitKey(0)

  #cv2.destroyAllWindows()