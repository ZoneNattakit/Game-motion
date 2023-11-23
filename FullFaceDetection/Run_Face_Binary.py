import os
import tensorflow as tf
import numpy as np
import cv2
from SetModelFunction import SetUpModel

def main() :
    setModel = SetUpModel()
    size = 224

    # Load face detection and face mask model
    path = r'Model'
    faceNet = cv2.dnn.readNet(os.path.join(path, 'face_detect', 'deploy.prototxt.txt'),
                            os.path.join(path, 'face_detect', 'res10_300x300_ssd_iter_140000.caffemodel'))

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (400, 400), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]

            # face_emotion = cv2.resize(face, (48, 48))
            # face_emotion = np.reshape(face_emotion, (1, 48, 144, 1))
            # face_emotion = np.reshape(face_emotion, (1, 144, 48, 1))
            # face_emotion = np.reshape(face_emotion, (3, 48, 48, 1)) / 255.0

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (size, size))
            face = np.reshape(face, (1, size, size, 3)) / 255.0
            

            result_mask = setModel.Mask_Detection(face)
            result_glasses = setModel.Glasses_Detection(face)
            # result_emotion = setModel.Emotion_Detection(face_emotion)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 1)
            cv2.rectangle(frame, (startX, startY - 50), (endX, startY), (0, 0, 255), -1)
            cv2.putText(frame, result_mask, (startX, startY - 30), 0, ((np.sqrt(((endX - startX)**2) + ((endY - startY) ** 2))/100) * 0.4), (255, 255, 255), 2, -1)
            cv2.putText(frame, result_glasses, (startX, startY - 10), 0, ((np.sqrt(((endX - startX)**2) + ((endY - startY) ** 2))/100) * 0.4), (255, 255, 255), 2, -1)
            # cv2.putText(frame, result_emotion, (startX, startY - 10), 0, ((np.sqrt(((endX - startX)**2) + ((endY - startY) ** 2))/100) * 0.4), (255, 255, 255), 2, -1)

        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', 800,600) # 1280,720
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()
main()