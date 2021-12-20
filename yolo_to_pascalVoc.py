
import cv2

f = open("data/ears/annotations/detection/train.txt", "r")

for x in f:
  y = x.split()
  x_min = y[1]
  y_min = y[2]
  w = y[3]
  h = y[4]

  img = cv2.imread("data/ears/annotations/detection/" + y[0])
  output="%s %s %s %s %s" % ('0', x_min, y_min, int(x_min)+int(w), int(y_min)+int(h))
  path = y[0].replace("png", "txt")

  f = open("data/ears/annotations/detection/" + path, "w")
  f.write(output)
  f.close()
