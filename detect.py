from flask import Flask, Response
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize
import threading
import time
import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate('/home/gttron/AgroProtect_Detection/agroprotect-e0049-firebase-adminsdk-ynhkq-10957fb18a.json')  # replace with your actual path
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://agroprotect-e0049-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

app = Flask(__name__)

# Global variables
frame = None
ref = db.reference('/detected_classes')

def generate():
    global frame
    while True:
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def update_firebase(labels):
    if labels:
        ref.child('label').set(', '.join(labels))
        ref.child('switch').set(True)
    else:
        ref.child('label').set("")
        ref.child('switch').set(False)

def run(model: str, max_results: int, score_threshold: float, camera_id: int, width: int, height: int) -> None:
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    COUNTER, FPS = 0, 0
    START_TIME = time.time()

    row_size = 50
    left_margin = 24
    text_color = (0, 0, 0)
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    detection_frame = None
    detection_result_list = []

    def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        nonlocal FPS, COUNTER, START_TIME, detection_frame

        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()
        detection_result_list.append(result)
        COUNTER += 1
        if result.detections:
            labels = [category.category_name for category in result.detections[0].categories
                      if category.category_name != 'human' and category.score > 0.7]
            if labels:
                threading.Thread(target=update_firebase, args=(labels,)).start()
            else:
                threading.Thread(target=update_firebase, args=(None,)).start()

    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           running_mode=vision.RunningMode.LIVE_STREAM,
                                           max_results=max_results, score_threshold=score_threshold,
                                           result_callback=save_result)
    detector = vision.ObjectDetector.create_from_options(options)

    while cap.isOpened():
        success, image = cap.read()
        image = cv2.resize(image, (640, 480))
        if not success:
            break

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        if detection_result_list:
            current_frame = visualize(current_frame, detection_result_list[0])
            detection_frame = current_frame
            detection_result_list.clear()

        if detection_frame is not None:
            global frame
            frame = detection_frame

        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Start Flask server
    flask_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000})
    flask_thread.daemon = True
    flask_thread.start()

    # Start object detection
    run(model='detect_metadata.tflite', max_results=5, score_threshold=0.25, camera_id=0, width=640, height=480)