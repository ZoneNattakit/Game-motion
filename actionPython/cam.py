from libraryAll import *
from makeLandmark import *

landMark = drawLandmark()

class vidCam():
    def __init__(self):
         pass
    def setMP(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            
            # Make detections
            with landMark.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                image, results = landMark.mediapipe_detection(frame, holistic)

            # Draw landmarks
            landMark.draw_styled_landmarks(image, results)
            

            # Show to screen
            cv2.imshow('OpenCV Feed', image)
            
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        landMark.draw_landmarks(frame, results)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return cap
    def get_actions(self):
        # You can collect the 'actions' variable here
            actions = np.array(['up', 'down', 'left', 'right']) #สามารถเพิ่มท่าทางได้อีก
            return actions

    def extract_keypoints(self, results):
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
            return np.concatenate([lh, rh])
        
    def collect_data(self):
        DATA_PATH = os.path.join('MP_Data') 
    
    # Actions that we try to detect
        actions = self.get_actions()

    # Thirty videos worth of data
        no_sequences = 10

    # Videos are going to be 30 frames in length
        sequence_length = 10

        for action in actions: 
            for sequence in range(no_sequences):
                try: 
                    os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
                except:
                    pass
        cap = cv2.VideoCapture(0)
            # Set mediapipe model 
        with landMark.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                # NEW LOOP
                # Loop through actions
                for action in actions:
                    # Loop through sequences aka videos
                    for sequence in range(no_sequences):
                        # Loop through video length aka sequence length
                        for frame_num in range(sequence_length):

                            # Read feed
                            ret, frame = cap.read()

                            # Make detections
                            image, results = landMark.mediapipe_detection(frame, holistic)
                            # Draw landmarks
                            landMark.draw_styled_landmarks(image, results)
                            
                            
                            # NEW Apply wait logic
                            if frame_num == 0: 
                                cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                                # Show to screen
                                cv2.imshow('OpenCV Feed', image)
                                cv2.waitKey(500)
                            else: 
                                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                                # Show to screen
                                cv2.imshow('OpenCV Feed', image)
                            
                            # NEW Export keypoints
                            keypoints = self.extract_keypoints(results)
                            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                            np.save(npy_path, keypoints)

                            # Break gracefully
                            if cv2.waitKey(10) & 0xFF == ord('q'):
                                break
                            
        cap.release()
        cv2.destroyAllWindows()

        label_map = {label:num for num, label in enumerate(actions)}

        sequences, labels = [], []
        for action in actions:
                for sequence in range(no_sequences):
                    window = []
                    for frame_num in range(sequence_length):
                        res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                        window.append(res)
                    sequences.append(window)
                    labels.append(label_map[action])
        
        return DATA_PATH, no_sequences, sequence_length, label_map
        