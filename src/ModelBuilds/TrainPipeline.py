from keras.models import Sequential, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import json
import os
from string import ascii_lowercase
from random import randint


class TrainPipeline:
    """
    # Train Pipeline
    
    Train pipeline will be used to instantly train a model on a data generator.
    
    Train pipeline takes in a sequential model as input and builds a model out of the sequential model and trains the model when train method is called.

    This stores all the evaluation data.
    """
    def __init__(self, model:Sequential, data, path:str, name:str=None, validData=None) -> None:
        """
        Initiates the pipeline
        
        Keyword arguments:

        model -- Sequential Model

        data -- Data to be used for training

        path -- Path to save the model

        name -- Name of the model

        Return: None
        """
        
        self.name = name
        self.createName()
        self.baseModel = model
        self.data = data
        self.validData = validData
        self.model = self.buildModel()
        self.history = {}
        self.path = path

    def createName(self):
        if self.name is None or self.name == "":
            self.name = "".join([ascii_lowercase[randint(0, len(ascii_lowercase)-1)] for _ in range(7)])

    def buildModel(self) -> Model:
        """
        Should be called by the __init__ method when initiating pipeline. Creates a model

        `Should be implemented by child`
        """
        return self.baseModel
    
    def historyCallback(self, history):
        """
        This method gets called whenever model is finished training
        
        Keyword arguments:

        history -- training history returned by `model.fit()`

        Return: None
        """
        self.history = history.history
        
    def train(self, epochs:int, optimizer="adam", loss="categorical_crossentropy", metrics:list=["accuracy"]):
        """
        Trains the model and saves the evaluation data
        
        Keyword arguments:

        epochs -- total number of training iterations

        optimizer -- optimizer for the model, `adam` by default

        loss -- loss function for training, `categorical cross entropy` by default

        metrics -- list of all the metrics to keep track of, keeps track of `accuracy` by default

        Return: None
        """
        self.model.compile(optimizer, loss, metrics)
        # compiling model 

        lrReduce = ReduceLROnPlateau(patience=5)
        checkPoint = ModelCheckpoint(os.path.join(self.path, f"{self.name}_checkpoint"), save_best_only=True)
        earlyStopping = EarlyStopping(patience=5)

        callbacks = [lrReduce, checkPoint, earlyStopping]

        if self.validData is None:
            history = self.model.fit(self.data, epochs=epochs, shuffle=False, callbacks=callbacks)
        else:
            history = self.model.fit(self.data, epochs=epochs, shuffle=False, validation_data=self.validData, callbacks=callbacks)
        # training model 

        self.historyCallback(history)
        # saving history 

    def save(self):
        """
        Saves the model and evaluation data as an h5 and json file
        """

        self.model.save(os.path.join(self.path, f"{self.name}.keras"))

        with open(os.path.join(self.path, f"{self.name}_eval.json"), "w") as f:
            json.dump(self.history, f)