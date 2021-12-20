from pathlib import PosixPath
import cv2

f = open("data/ears/annotations/detection/test.txt", "r")

for x in f:
  y = x.split()
  x_min = y[1]
  y_min = y[2]
  w = y[3]
  h = y[4]

  img = cv2.imread("data/ears/annotations/detection/" + y[0])
  im_h = img.shape[0]
  im_w = img.shape[1]
  print(im_h, im_w)
  x_center = (2 * int(x_min) + int(w))/2/im_w
  y_center = (2 * int(y_min) + int(h))/2/im_h
  ww = int(w)/im_w
  hh = int(h)/im_h

  output="%s %s %s %s %s" % ('0', x_center, y_center, ww, hh)
  path = y[0].replace("png", "txt")

  f = open("data/ears/annotations/detection/" + path, "w")
  f.write(output)
  f.close()

  
