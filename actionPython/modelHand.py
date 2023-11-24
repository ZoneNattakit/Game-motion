from libraryAll import *
from cam import vidCam
from makeLandmark import *
from sklearn.metrics import multilabel_confusion_matrix 
# import pyautogui
import ctypes
import pygetwindow as gw
import time

landMark = drawLandmark()

class modelCam:
    def __init__(self):
        self.model = None
    
    def load_data(self, DATA_PATH, no_sequences, sequence_length):
        sequences, labels = [], []
        actions = self.get_actions()
        label_map = {label:num for num, label in enumerate(actions)}

        for action in actions:
            for sequence in range(no_sequences):
                window = []
                for frame_num in range(sequence_length):
                    res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(label_map[action])

        X = np.array(sequences)  # X should be a 2D array of sequences
        y = to_categorical(labels)  # One-hot encode labels

        return X, y

    def train_model(self, X, y, actions):
        actions = self.get_actions()
        print(type(actions))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
        DATA_PATH = 'MP_Data'
        no_sequences = 30
        sequence_length = 30
        
        X, y = self.load_data(DATA_PATH, no_sequences, sequence_length)

        model = Sequential()
        model.add(Conv1D(256, kernel_size=3, activation='sigmoid', input_shape=(sequence_length, 126)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(128, kernel_size=3, activation='sigmoid'))
        model.add(Flatten())
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(256, activation='sigmoid'))
        model.add(Dense(actions.shape[0], activation='softmax'))

        model.compile(optimizer='Adam', loss=tf.losses.MeanSquaredError(), metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=1000, batch_size=32)

        return model
    
    def savemodel(self, model):
            tf.keras.models.save_model(model, 'action.h5')
            return model
    def loading_model(self, model):
            model = tf.keras.models.load_model('action.h5')
            return model
        
    def get_actions(self):
        # You can collect the 'actions' variable here
            actions = np.array(['up', 'down', 'left', 'right'])
            return actions
        
    def press_key(self, key_code):
        ctypes.windll.user32.keybd_event(key_code, 0, 0, 0)
        time.sleep(0.025)
        ctypes.windll.user32.keybd_event(key_code, 0, 2, 0)
        
    def predict_and_res(self, X_test, y_test):
        actions = self.get_actions()

        cap = cv2.VideoCapture(0)
        with landMark.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            model = self.loading_model('action.h5')
            sequence = []
            sentence = []
            predictions = []
            threshold = 0.5
            res = [0.7, 0.2, 0.1]
            last_action = None  # Variable to store the last predicted action
            prediction_enabled = True  # Toggle for checking predictions
            hand_detected = False  # Flag to track hand detection

            LEFT_ARROW = 0x25
            UP_ARROW = 0x26
            RIGHT_ARROW = 0x27
            DOWN_ARROW = 0x28
            last_update_time = time.time()  # Initialize last update time

            while cap.isOpened():
                res, frame = cap.read()
                image, results = landMark.mediapipe_detection(frame, holistic)

                # Check if hands are detected
                if hand_detected == False:
                    keypoints = vidCam.extract_keypoints(self, results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    # Hand is detected
                    hand_detected = True

                    if len(sequence) == 30 and prediction_enabled:
                        input_sequence = np.expand_dims(sequence, axis=0)
                        res = model.predict(input_sequence)[0]

                        predicted_action = actions[np.argmax(res)]

                        # Check if the predicted action is different from the last action
                        if predicted_action != last_action:
                            if predicted_action == 'left':
                                self.press_key(LEFT_ARROW)
                                last_action = 'left'
                                print(predicted_action)

                            elif predicted_action == 'right':
                                self.press_key(RIGHT_ARROW)
                                last_action = 'right'
                                print(predicted_action)

                            elif predicted_action == 'up':
                                self.press_key(UP_ARROW)
                                last_action = 'up'
                                print(predicted_action)

                            elif predicted_action == 'down':
                                self.press_key(DOWN_ARROW)
                                last_action = 'down'
                                print(predicted_action)

                        predictions.append(np.argmax(res))

                        if np.unique(predictions[-1:])[0] == np.argmax(res):
                            if res[np.argmax(res)] > threshold:
                                if len(sentence) > 0:
                                    if actions[np.argmax(res)] != sentence[-1]:
                                        sentence.append(actions[np.argmax(res)])
                                else:
                                    sentence.append(actions[np.argmax(res)])

                        if hand_detected:
                            current_time = time.time()

                            # Display the message for 1 second
                            if current_time - last_update_time < 1:
                                resized_image = cv2.resize(image, (320, 240))
                                cv2.rectangle(resized_image, (0, 0), (320, 40), (245, 117, 16), -1)
                                cv2.putText(resized_image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                            cv2.LINE_AA)

                                cv2.imshow('OpenCV Feed', resized_image)
                            else:
                                # Clear the sentence after 1 second
                                sentence = []
                                last_update_time = current_time

                    else:
                        # No hand detected, reset message and flag
                        hand_detected = False
                        sentence = []


                else:
                    # No hand detected, reset message and flag
                    hand_detected = False
                    sentence = []

                # Display message only if hand is detected
                if hand_detected:
                    resized_image = cv2.resize(image, (320, 240))
                    cv2.rectangle(resized_image, (0, 0), (320, 40), (245, 117, 16), -1)
                    cv2.putText(resized_image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

                    cv2.imshow('OpenCV Feed', resized_image)

                key = cv2.waitKey(10)
                if key & 0xFF == ord('q'):
                    break
                elif key == ord('t'):  # Toggle prediction
                    prediction_enabled = not prediction_enabled

            cap.release()
            cv2.destroyAllWindows()
