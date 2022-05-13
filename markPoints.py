# importing the module
import cv2
import numpy as np
  
# function to detect the coordinates of danger zone
# click on the 4 points and press space bar to exit
def click_event(event, x, y, flags, pts):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x,y])
        cv2.putText(img, '.', (x,y), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 5)
        cv2.imshow('image', img)
    return pts
 
# driver function
if __name__=="__main__":

    # code to get the first frame of the input video
    vidcap = cv2.VideoCapture('solidWhiteRight.mp4')
    success,image = vidcap.read()
    cv2.imwrite("frame1.jpg", image)

    # reading the image
    img = cv2.imread('frame1.jpg', 1)
 
    # displaying the image
    cv2.imshow('image', img)

    # list to get the coordinates of danger zone
    pts = []

    # setting mouse handler for the image 
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event,pts)
    
    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()

    # path
    path = r'frame1.jpg'
    
    # Reading an image in default mode
    image = cv2.imread(path)
    
    # Polygon corner points coordinates
    pts = np.array(pts, np.int32)
    
    # Using cv2.polylines() method
    image = cv2.polylines(image, [pts], True, (0, 0, 255), 2)

    cv2.imshow('image', image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()