from imageai.Detection import VideoObjectDetection
#-------------------------------
vid_obj_detect = VideoObjectDetection()
vid_obj_detect.setModelTypeAsYOLOv3()
#-------------------------------
vid_obj_detect.setModelPath(r"E:/dataset_vehicle_detection/yolo.h5")
vid_obj_detect.loadModel()
#--------------------------------
detected_vid_obj = vid_obj_detect.detectObjectsFromVideo(
    input_file_path =  r"E:/dataset_vehicle_detection/input_video.mp4",
    output_file_path = r"E:/dataset_vehicle_detection/output_video",
    frames_per_second=15,
    log_progress=True,
    return_detected_frame = True
)