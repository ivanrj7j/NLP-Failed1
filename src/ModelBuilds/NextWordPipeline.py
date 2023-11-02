from uuid import uuid1
from keras.models import Model, Sequential
from ModelBuilds.TrainPipeline import TrainPipeline
from src.DataLoader import TrainLoader
from ModelBuilds.funcs import createModel

class NextWordPipeline(TrainPipeline):
    """
    Used to train next word models
    """
    def __init__(self, model: Sequential, data: TrainLoader, name: str = str(uuid1), embeddingOutput:int=500) -> None:
        super().__init__(model, data, name)
        self.trainLoader = data
        self.data = data.getTensorflowDataset()
        self.embeddingOutput = embeddingOutput

    def buildModel(self) -> Model:
        """"
        Builds a new next word model with input layer as vocab length, an embedding, sequential model, output layer
        """
        return createModel(self.trainLoader.sequenceLength, self.trainLoader.tokenizer.tokenizer.get_vocab_size(), self.embeddingOutput, self.model, self.name)
        
    