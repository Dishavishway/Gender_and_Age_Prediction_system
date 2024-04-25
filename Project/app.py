import cv2
import numpy as np
from flask import Flask
from flask import Flask, render_template,Response, request
from PIL import Image
import subprocess
import io

app = Flask(__name__)

def getFaceBox(net, frame, conf_threshold=0.75):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

    return frameOpencvDnn, bboxes

def go_live_and_detect():
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"

    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"

    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)','(20-25)','(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # Load the network
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Set desired frame width and height
    width = 1280
    height = 720

    # Set the frame size manually
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    padding = 20

    while cv2.waitKey(1) < 0:
        # Read frame
        hasFrame, frame = cap.read()

        if not hasFrame:
            cv2.waitKey()
            break

        # Resize frame to desired size
        resized_frame = cv2.resize(frame, (width, height))

        # Face detection and processing
        frameFace, bboxes = getFaceBox(faceNet, resized_frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            continue
        for bbox in bboxes:
            face = resized_frame[max(0, bbox[1] - padding):min(bbox[3] + padding, resized_frame.shape[0] - 1),
                max(0, bbox[0] - padding):min(bbox[2] + padding, resized_frame.shape[1] - 1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print("Age Output : {}".format(agePreds))
            print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            label = "{},{}".format(gender, age)
            cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow("Age Gender Demo", frameFace)

    cv2.destroyAllWindows()
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

# Route to photo.html
# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
    # if request.method == 'POST':
        # f = request.files['fileToUpload'].read()
        # img = Image.open(io.BytesIO(f))
        # img_ip = np.asarray(img, dtype="uint8")
        # print(img_ip)
        # return Response(gen_frames_photo(img_ip), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live')
def video_feed():
    return Response(go_live_and_detect(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')


if __name__ == '__main__':
    app.run(debug=True)