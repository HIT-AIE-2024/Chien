import os
from script.yolo_detector import YOLODetector
from config.configs import IMG_PATH, VIDEO_PATH

detector = YOLODetector()
file_path = os.path.join(IMG_PATH, 'people.jpg')
# file_path = ''

if not os.path.isfile(file_path):
    print("Camera on")
    detector.detect_camera()
else:
    file_extension = os.path.splitext(file_path)[-1].lower()

    if file_extension in ['.jpg', '.jpeg', '.png']:
        detector.detect_image(file_path)
    elif file_extension in ['.mp4', '.avi', '.mkv']:
        detector.detect_video(file_path)
    else:
        print("Unsupported file type. Please provide a valid image or video file.")
