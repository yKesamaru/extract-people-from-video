import sys
sys.path.append('/usr/lib/python3/dist-packages')
import cv2
import numpy as np

# 黒い画像を作成
image = np.zeros((500, 500, 3), dtype="uint8")

# 画像を表示
cv2.imshow('Test Window', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
