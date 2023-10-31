from keras.layers import Input, Embedding, Dense
from keras.models import Sequential, Model

def createModel(sequenceLength:int, totalTokens:int, embeddingOutput:int, hiddenLayer:Sequential, name:str) -> Model:
    inputLayer = Input(sequenceLength, name="input_layer")

    embeddingLayer = Embedding(totalTokens, embeddingOutput, name="embedding_layer")(inputLayer)
    
    hiddenLayer = hiddenLayer(embeddingLayer)

    outputLayer = Dense(totalTokens, activation="softmax", name="output")(hiddenLayer)

    return Model(inputs=inputLayer, outputs=outputLayer, name=name)

