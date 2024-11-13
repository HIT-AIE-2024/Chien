import cv2
import os
from ultralytics import YOLO
from config.configs import MODEL_PATH, IMG_OUTPUT_PATH, VID_OUTPUT_PATH

class YOLODetector:
    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the YOLODetector with a specified YOLO model.

        Args:
            model_path (str, optional): Path to the YOLO model. Defaults to MODEL_PATH.
        """
        try:
            self.model = YOLO(model=model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")


    def detect_image(self, img_path: str) -> None:
        """
        Perform object detection on an image.

        Args:
            img_path (str): Path to the input image.
        """
        image = cv2.imread(img_path)
        if image is None:
            print(f"Image not found at: {img_path}")
            return

        results = self.model(image)
        annotated_image = results[0].plot()

        self.save_image(img_path, annotated_image)

        cv2.imshow('YOLOv8 Image Detection', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, img_path: str, annotated_image) -> None:
        """
        Save the annotated image.

        Args:
            img_path (str): Path to the original input image.
            annotated_image: The image with bounding boxes and labels.
        """
        input_filename = os.path.basename(img_path)
        file_name, file_extension = os.path.splitext(input_filename)

        os.makedirs(IMG_OUTPUT_PATH, exist_ok=True)

        output_img_path = os.path.join(IMG_OUTPUT_PATH, f"{file_name}_output{file_extension}")
        cv2.imwrite(output_img_path, annotated_image)
        print(f"Image saved at: {output_img_path}")

    def detect_video(self, video_path: str) -> None:
        """
        Perform object detection on a video.

        Args:
            video_path (str): Path to the input video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video at: {video_path}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        input_filename = os.path.basename(video_path)
        file_name, _ = os.path.splitext(input_filename)

        os.makedirs(VID_OUTPUT_PATH, exist_ok=True)
        output_video_path = os.path.join(VID_OUTPUT_PATH, f"{file_name}_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame)
                annotated_frame = results[0].plot()

                out.write(annotated_frame)

                cv2.imshow('YOLOv8 Video Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('e'):
                    break

        except Exception as e:
            print(f"Error processing video: {e}")
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        print(f"Video saved at: {output_video_path}")

    def detect_camera(self) -> None:
        """
        Perform real-time object detection using the camera.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot access the camera")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Cannot read frame")
                    break

                results = self.model(frame)
                annotated_frame = results[0].plot()

                cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('e'):
                    break
        except Exception as e:
            print(f"Error with camera detection: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()