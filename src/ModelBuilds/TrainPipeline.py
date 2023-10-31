from keras.models import Sequential, Model
import json
import os
from uuid import uuid1

class TrainPipeline:
    """
    # Train Pipeline
    
    Train pipeline will be used to instantly train a model on a data generator.
    
    Train pipeline takes in a sequential model as input and builds a model out of the sequential model and trains the model when train method is called.

    This stores all the evaluation data.
    """
    def __init__(self, model:Sequential, data, name:str=str(uuid1())) -> None:
        """
        Initiates the pipeline
        
        Keyword arguments:

        model -- Sequential Model

        data -- Data to be used for training

        name -- Name of the model

        Return: None
        """
        
        self.baseModel = model
        self.data = data
        self.model = self.buildModel()
        self.history = {}
        self.name = name

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

        history = self.model.fit(self.data, epochs=epochs, shuffle=False)
        # training model 

        self.historyCallback(history)
        # saving history 

    def save(self, path:str):
        """
        Saves the model and evaluation data as an h5 and json file
        
        Keyword arguments:

        path -- path to saving the model

        Return: None
        """

        self.model.save(os.path.join(path, f"{self.name}.h5"))
        json.dump(self.history, os.path.join(path, f"{self.name}_eval.json"))