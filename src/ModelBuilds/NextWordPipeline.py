from keras.models import Model, Sequential
from ModelBuilds.TrainPipeline import TrainPipeline
from DataLoader import TrainLoader
from ModelBuilds.funcs import createModel

class NextWordPipeline(TrainPipeline):
    """
    Used to train next word models
    """
    def __init__(self, model: Sequential, data: TrainLoader, name: str = "", embeddingOutput:int=500, validData:TrainLoader=None) -> None:
        self.trainLoader = data
        self.embeddingOutput = embeddingOutput
        super().__init__(model, data, name)
        self.data = data.getTensorflowDataset()
        if validData is not None:
            self.validData = validData.getTensorflowDataset()

    def buildModel(self) -> Model:
        """"
        Builds a new next word model with input layer as vocab length, an embedding, sequential model, output layer
        """
        return createModel(self.trainLoader.sequenceLength, self.trainLoader.tokenizer.tokenizer.get_vocab_size(), self.embeddingOutput, self.baseModel, self.name)
        
    