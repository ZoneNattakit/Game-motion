import tensorflow as tf
import os
import numpy as np
from deepface import DeepFace

class SetUpModel :
    def __init__(self) :
        path = "Model"
        self.model_mask = tf.keras.models.load_model(os.path.join(path, 'face_mask.h5'))
        self.model_glasses = tf.keras.models.load_model(os.path.join(path, 'face_glasses.h5'))
        self.model_emotion = DeepFace.build_model("Emotion")
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def Mask_Detection(self, data) :
        result = np.argmax( self.model_mask.predict(data))
        if result == 0:
            return "Mask"
        else:
            return "No Mask"
    
    def Glasses_Detection(self, data) :
        result = np.argmax( self.model_glasses.predict(data))
        if result == 0:
            return "Glasses"
        else:
            return "No Glasses"

    def Emotion_Detection(self, data) :
        result = self.model_emotion.predict(data)[0]
        emotion_idx = result.argmax()
        return self.emotion_labels[emotion_idx]
