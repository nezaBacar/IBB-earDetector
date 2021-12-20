import cv2
import numpy as np
from sklearn import metrics
from podm.podm import get_pascal_voc_metrics, BoundingBox, MetricPerClass
import matplotlib.pyplot as plt

class Evaluation:

    def convert2mask(self, mt, shape):
        # Converts coordinates of bounding-boxes into blank matrix with values set where bounding-boxes are.

        t = np.zeros([shape, shape])
        for m in mt:
            x, y, w, h = m
            cv2.rectangle(t, (x,y), (x+w, y+h), 1, -1)
        return t

    def prepare_for_detection(self, prediction, ground_truth):
            # For the detection task, convert Bounding-boxes to masked matrices (0 for background, 1 for the target). If you run segmentation, do not run this function

            if len(prediction) == 0:
                return [], []

            # Large enough size for base mask matrices:
            shape = 2*max(np.max(prediction), np.max(ground_truth)) 
            
            p = self.convert2mask(prediction, shape)
            gt = self.convert2mask(ground_truth, shape)

            return p, gt

    def iou_compute(self, p, gt):
            # Computes Intersection Over Union (IOU)
            if len(p) == 0:
                return 0

            intersection = np.logical_and(p, gt)
            union = np.logical_or(p, gt)

            iou = np.sum(intersection) / np.sum(union)

            return iou

    def mAP(self, annot_list, p_list, im_name, iou):
            #ssd in right form
            gt = []
            p = []
            if annot_list:
                for el in annot_list:
                    element = BoundingBox(im_name, "ear", el[0],el[1],el[0]+el[2],el[1]+el[3], iou)
                    gt.append(element)
            if p_list:
                for el in p_list:
                    element = BoundingBox(im_name, "ear", el[0],el[1],el[2],el[3], el[4])
                    p.append(element)
            results = get_pascal_voc_metrics(gt, p, .75)

            ap = results["ear"].get_mAP(results)
            return ap

    def plot_histogram(selr, arr, title, value, labels):
            x = np.arange(len(arr))
            plt.bar(x, height=arr)
            #plt.bar(x_pos, height, color = (0.5,0.1,0.5,0.6))
            plt.xticks(x, labels)
            plt.ylim(0, 1)
            plt.title(title)
            plt.ylabel(value)
            plt.show()
        