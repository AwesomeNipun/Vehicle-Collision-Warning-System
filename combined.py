import cv2 as cv
import numpy as np
from sympy import Point, Polygon
from kalmanfilter import KalmanFilter

def findObjects(outputs,img, video, pts):
    global j
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    pts = np.array(pts, np.int32)
    # print(pts)

    kf = KalmanFilter()     # kalman filter
    for i in indices:
        box = bbox[i]
        # print(box)
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,x+w,y+h)
        c_x = int(x+w/2)
        c_y = int(y+h/2)
        # print(c_x, c_y)
        p_x, p_y = kf.predict(c_x, c_y)
        cv.rectangle(img, (x, y), (x+w,y+h), (204, 0, 0), 2)
        out = cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (204, 0, 0), 2)
        x = int(p_x - w/2)
        y = int(p_y - h/2)
        arr = [[x,y], [x+w, y], [x+w, y+h], [x, y+h]]
        #-------------------------------------------------------------------
        
        # creating points using Point()
        p1, p2, p3, p4 = map(Point, pts)
        p5, p6, p7,p8 = map(Point, arr)
        
        # creating polygons using Polygon()
        poly1 = Polygon(p1, p2, p3, p4)
        poly2 = Polygon(p5, p6, p7,p8)
        
        # using intersection()
        isIntersection = poly1.intersection(poly2)
        if not(isIntersection==[]):
            cv.putText(img, "WARNING!", (int(frame_width/2)-100, 100), cv.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2 )
            

    #----------------------------------------------

    
    # Using cv2.polylines() method
    image = cv.polylines(out, [pts], True, (0, 255, 255), 2)

    #--------------------------------------------------            
    video.write(image)


  
# function to detect the coordinates of danger zone
# click on the 4 points and press space bar to exit
def click_event(event, x, y, flags, pts):
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        pts.append([x,y])
        cv.putText(img_, '.', (x,y), cv.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 5)
        cv.imshow('image', img_)
    return pts

 
# driver function
if __name__=="__main__":

    # code to get the first frame of the input video
    vidcap = cv.VideoCapture('sample_1.mp4')
    #--------------------------------------------------
    whT = 320
    confThreshold =0.5
    nmsThreshold= 0.2

    #### LOAD MODEL
    ## Coco Names
    classesFile = "coco.names"
    classNames = []
    with open(classesFile, 'rt') as f:
        classNames = f.read().split('\n')
    # print(classNames)
    ## Model Files
    modelConfiguration = "yolov3.cfg.txt"
    modelWeights = "yolov3.weights"
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    
    #-----------------------------------------------------
    success,image_ = vidcap.read()
    cv.imwrite("frame1.jpg", image_)

    # reading the image
    img_ = cv.imread('frame1.jpg', 1)
    cv.putText(img_, 'Select 4 points to mark danger zone',(20,50), cv.FONT_HERSHEY_DUPLEX, 1, (153, 51, 255), 2)
    cv.putText(img_, 'Press any key to continue',(20,110), cv.FONT_HERSHEY_DUPLEX, 1, (153, 51, 255), 2)
    # displaying the image
    cv.imshow('image', img_)

    # list to get the coordinates of danger zone
    pts = []

    # setting mouse handler for the image 
    # and calling the click_event() function
    cv.setMouseCallback('image', click_event,pts)
    
    # wait for a key to be pressed to exit
    cv.waitKey(0)

    # close the window
    cv.destroyAllWindows()

    # path
    path = r'frame1.jpg'
    
    # Reading an image in default mode
    image_ = cv.imread(path)
    
    # Polygon corner points coordinates
    pts = np.array(pts, np.int32)
    
    # Using cv2.polylines() method
    # image_ = cv.polylines(image_, [pts], True, (0, 0, 255), 2)

    # cv.imshow('image', image_)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

    #---------------------------------------------------------
    # process to get video output
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))
    frame_size = (frame_width,frame_height)
    fps = int(vidcap.get(cv.CAP_PROP_FPS))
    vid_out = cv.VideoWriter("output.avi", cv.VideoWriter_fourcc('M','J','P','G'), fps, frame_size )
    while True:
        success, img = vidcap.read()
        # print("detecting...")
        try:
            blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
            net.setInput(blob)
            layersNames = net.getLayerNames()
            outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
            outputs = net.forward(outputNames)
            findObjects(outputs,img, vid_out, pts)
            cv.putText(img, "Press Esc to exit", (20, 50), cv.FONT_HERSHEY_DUPLEX, 1, (102, 255, 102), 2)
            cv.imshow("Image", img) 
            val = cv.waitKeyEx(1)
            if val == 27:
                cv.destroyAllWindows()
                break
            
        except:
            break
    #---------------------------------------------------------

