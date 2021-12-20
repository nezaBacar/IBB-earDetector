import cv2, sys, os, numpy as np

class Detector:
	# This example of a detector detects faces. However, you have annotations for ears!
	cascade_l = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_leftear.xml'))
	cascade_r = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_rightear.xml'))
  
	def detect(self, img):
		r_list = self.detect_r(img)
		l_list = self.detect_l(img)

		if (not isinstance(r_list, tuple) and not isinstance(l_list, tuple)):
			return np.concatenate((r_list, l_list), axis=0)
		elif (not isinstance(r_list, tuple) and isinstance(l_list, tuple)):
		  return r_list
		return l_list

	def detect_r(self, img):
		det_list_r = self.cascade_r.detectMultiScale(img, 1.05, 1)
		return det_list_r

	def detect_l(self, img):
		det_list_l = self.cascade_l.detectMultiScale(img, 1.05, 1)
		return det_list_l

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	detector = CascadeDetector()
	detected_loc = detector.detect_r(img)
	for x, y, w, h in detected_loc:
		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
	cv2.imwrite(fname + '.detected.jpg', img)