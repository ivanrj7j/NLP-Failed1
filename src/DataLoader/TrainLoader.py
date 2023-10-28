import numpy as np
from DataLoader import Paginator
from Tokenizer import NLPTokenizer

class TrainLoader(Paginator):
    """
    # TrainLoader

    TrainLoader is used for loading data for training NLP models, this module is a child of `Paginator`

    TrainLoader reads through a pandas iterable and tokenizes text using `NLPTokenizer`
    """
    
    def __init__(self, batchSize: int, sequenceLength:int, tokenizerFile:str, shouldShuffle=True) -> None:
        """
        Initiates module
        
        Keyword arguments:

        batchSize -- number of items in the batch

        tokenizerFile (str) -- Path to the tokenizer json file

        sequenceLength (int) -- Maximum tokens

        shouldShuffle -- shuffles data if true

        Return: None
        """
        
        super().__init__(batchSize, shouldShuffle)
        self.tokenizer = NLPTokenizer(tokenizerFile, sequenceLength)
        self.sequenceLength = sequenceLength

    def nextBatch(self) -> np.ndarray:
        return super().nextBatch()
    
    def trainTokenizeText(self, text:str) -> np.ndarray:
        """
        Tokenizes a text for passing through training 
        
        Keyword arguments:

        text -- text to be tokenized

        Return: a matrix of tokens
        """
        
        encodedMatrix = self.tokenizer.encode(text, False)
        x = []
        y = []
        for vector in encodedMatrix:
            for i in range(1, len(vector)):
                start = max(0, i-self.sequenceLength)
                data = vector[start:i]
                if len(data) < self.sequenceLength:
                    data = np.pad(data, (self.sequenceLength-len(data), 0), 'constant', constant_values=-1)
                x.append(data)
                y.append(vector[i])
        
        return np.vstack(x).astype(np.int32), np.array(y, dtype=np.int32)
        
