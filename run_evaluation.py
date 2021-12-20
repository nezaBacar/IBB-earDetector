import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def get_annotations(self, annot_name):
            with open(annot_name) as f:
                lines = f.readlines()
                annot = []
                for line in lines:
                    l_arr = line.split(" ")[1:5]
                    l_arr = [int(i) for i in l_arr]
                    annot.append(l_arr)
            return annot
    
    def evaluate(self, eval, annot_list, pred_list_short, pred_list, im_name):
        p, gt = eval.prepare_for_detection(pred_list_short, annot_list)
        iou = eval.iou_compute(p, gt)
        if pred_list:
            ap = eval.mAP(annot_list, pred_list, im_name, iou)
        else:
            ap = 0
        return [iou, ap]

    def run_evaluation(self):
        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []
        ap_arr = []
        iou_arr1 = []
        ap_arr1 = []
        iou_arr2 = []
        ap_arr2 = []
        iou_arr3 = []
        ap_arr3 = []
        iou_arr4 = []
        ap_arr4 = []
        preprocess = Preprocess()
        eval = Evaluation()
        
        import detectors.cascade_ear_detector.ear_detector as cascade_ear_detector
        cascade_ear_detector = cascade_ear_detector.Detector()
        import detectors.yolo3_ear_detector.ear_detector as yolo3_ear_detector
        yolo3_ear_detector = yolo3_ear_detector.Detector()
        import detectors.yolo3tiny_ear_detector.ear_detector as yolo3tiny_ear_detector
        yolo3tiny_ear_detector = yolo3tiny_ear_detector.Detector()
        import detectors.yolo4tiny_ear_detector.ear_detector as yolo4tiny_ear_detector
        yolo4tiny_ear_detector = yolo4tiny_ear_detector.Detector()
        import detectors.mobilenet_ear_detector.ear_detector as mobilenet_ear_detector
        mobilenet_ear_detector = mobilenet_ear_detector.Detector()
        
        for im_name in im_list:
            # Read an image
            img = cv2.imread(im_name)
            img_grayscale = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)

            # Apply some preprocessing
            # img1 = preprocess.histogram_equlization_rgb(img) 
            # img2 = preprocess.increase_brightness(img) 
            # img3 = preprocess.sharpen(img) 
            # img_grayscale = preprocess.histogram_equlization_g(img_grayscale) 

            pr = []
            # Run the detector. It runs a list of all the detected bounding-boxes.
            prediction_list_cascade = cascade_ear_detector.detect(img)
            pr.append(prediction_list_cascade)
            print(prediction_list_cascade)
            prediction_list_cascade1, all_pred_1 = yolo3_ear_detector.detect(img, im_name)
            pr.append(prediction_list_cascade1)
            prediction_list_cascade2, all_pred_2 = yolo3tiny_ear_detector.detect(img, im_name)
            pr.append(prediction_list_cascade2)
            prediction_list_cascade3, all_pred_3 = yolo4tiny_ear_detector.detect(img, im_name)
            pr.append(prediction_list_cascade3)
            prediction_list_cascade4 = mobilenet_ear_detector.detect(im_name)
            pr.append(prediction_list_cascade4)

            # prediction_list_cascade, all_pred = cascade_ear_detector.detect(img_grayscale, im_name)

            # Read annotations:
            annot_name = os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt'
            annot_list = self.get_annotations(annot_name)
            
            # Evaluate all detectors
            iou, ap = self.evaluate(eval, annot_list, prediction_list_cascade, None, im_name)
            iou_arr.append(iou)

            iou1, ap1 = self.evaluate(eval, annot_list, prediction_list_cascade1, all_pred_1, im_name)
            iou_arr1.append(iou1)
            ap_arr1.append(ap1)
            
            iou2, ap2 = self.evaluate(eval, annot_list, prediction_list_cascade2, all_pred_2, im_name)
            iou_arr2.append(iou2)
            ap_arr2.append(ap2)

            iou3, ap3 = self.evaluate(eval, annot_list, prediction_list_cascade3, all_pred_3, im_name)
            iou_arr3.append(iou3)
            ap_arr3.append(ap3)

            iou4, ap4 = self.evaluate(eval, annot_list, prediction_list_cascade4, None, im_name)
            iou_arr4.append(iou4)
            
        
        miou_arr = []
        miou = np.average(iou_arr)
        miou_arr.append(miou)
        miou1 = np.average(iou_arr1)
        miou_arr.append(miou1)
   
        miou2 = np.average(iou_arr2)
        miou_arr.append(miou2)
        miou3 = np.average(iou_arr3)
        miou_arr.append(miou3)
        miou4 = np.average(iou_arr4)
        miou_arr.append(miou4)
        eval.plot_histogram(miou_arr, "All models", "Mean IOU", ["VJ", "YoloV3", "YoloV3tiny", "YoloV4tiny", "MobileNet"])
     
        map_arr = []
        mAP1 = np.average(ap_arr1)
        map_arr.append(mAP1)
        mAP2 = np.average(ap_arr2)
        map_arr.append(mAP2)
        mAP3 = np.average(ap_arr3)
        map_arr.append(mAP3)
        print(ap_arr1)
        print(ap_arr2)
        print(ap_arr3)
        eval.plot_histogram(map_arr, "Yolo models", "mAP", ["YoloV3", "YoloV3tiny", "YoloV4tiny",])

if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()