import os
import cv2

path_img = r'/images'
dirs = os.listdir(path_img)
os.chdir(path_img)
try:
    os.mkdir('rename')
except:
    pass
count = 0
for img in dirs:
    img = cv2.imread(img, 1)
    Str = "%06d" % count
    path = os.path.join('rename', Str + '.jpg')
    cv2.imwrite(path, img)
    count += 1
