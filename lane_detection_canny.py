import cv2
import numpy as np

path = 'data/clip-video.mp4'

cap = cv2.VideoCapture(path)

while (cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        resize = cv2.resize(frame, (854, 480))
        
        # crop image
        resize = resize[230:355, 50:804]

        # convert to gray
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

        # gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # canny
        edge = cv2.Canny(blur, 150, 200)

        edge3C = np.zeros( (np.array(edge).shape[0], np.array(edge).shape[1], 3) ).astype(np.uint8)
        edge3C[:,:,0] = edge
        edge3C[:,:,1] = edge
        edge3C[:,:,2] = edge

        images = np.vstack((resize, edge3C))

        cv2.imshow('frame', images)

        key = cv2.waitKey(30)

        if key == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()