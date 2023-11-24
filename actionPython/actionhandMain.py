from cam import *
import os
from makeLandmark import *
from modelHand import modelCam
from writeLogs import LoggingFunction


model_train = modelCam()
captureVid = vidCam()
Logger = LoggingFunction('actionhandMain')

def main():

    model_path = "action.h5"
    if os.path.isfile(model_path):
        actions = captureVid.get_actions()
        DATA_PATH = "MP_Data"
        no_sequences = 30
        sequence_length = 30
    
        # Load test data
        X_test, y_test = model_train.load_data(DATA_PATH, no_sequences, sequence_length)  
        
        if X_test is not None and y_test is not None:
            model_train.predict_and_res(X_test, y_test)
        else:
            print("X_test and y_test is empty.")
    else:       
        actions = vidCam.get_actions
        DATA_PATH = "MP_Data"
        no_sequences = 30
        sequence_length = 30
        captureVid.collect_data()

        X, y = model_train.load_data(DATA_PATH, no_sequences, sequence_length)
        conv_model = model_train.train_model(X, y, actions)
        model_train.savemodel(conv_model)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        Logger.logErrorMessage(e, True)