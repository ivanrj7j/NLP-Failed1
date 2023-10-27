import numpy as np
from DataLoader.Errors import PrmotionError

class Paginator:
    """
    # Paginator

    Paginator holds tensor of given shape let's say `(m, n)`. Here `m` is the batchsize and `n` is the size of the vector. It acts as an iterator, only returning matrix of height `n` and storing remaining data to be accessed later.

    ## How it works?

    This class will be used as a parent for other classes, and each child should  have a `nextBatch` method, `__next__` method wil use the `nextBatch` method to get data and store the later data. If the nextBatch returns data with size less than `m`, then the `nextBatch` method will be called again until batch size >= `m`
    """
    def __init__(self, batchSize:int) -> None:
        """
        Initiates the object
        
        Keyword arguments:

        batchSize -- number of items in the batch

        Return: Initiates the object
        """
        self.batchSize = batchSize
        # setting the shape 

        self.cache1:np.ndarray = np.ndarray((0))
        # stores data of maximum size batchSize 
        self.cache2:np.ndarray = np.ndarray((0))
        # can store data of any size 

        # initiating cache 

    def nextBatch(self) -> np.ndarray:
        """
        This method should be implemented by the child and return a numpy array
        """
        raise NotImplementedError("This method should be implemented by the child and return a numpy array")
    

    def promoteData(self, a:np.ndarray, b:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Promotes data from one level to higher
        
        Keyword arguments:

        a -- Numpy array in which data is promoted `to` 

        b -- Numpy array in which data is promoted `from` 

        Return: Returns updated `a` and `b`
        """
        
        aSize = a.shape[0]  
        bSize = b.shape[0]

        if bSize == 0:
            raise PrmotionError("Nothing left to promote")
        
        if aSize==self.batchSize:
            return a, b

        if aSize < self.batchSize and bSize > 0:
            removeChunk = self.batchSize-aSize if (bSize > self.batchSize-aSize) else bSize
            updateData = b[:removeChunk]
            # getting new data to be pushed to a 

            b = b[removeChunk:]
            # updating b

            a = np.vstack((updateData, a))

        elif aSize > self.batchSize:
            removeChunk = aSize-self.batchSize-1
            updateData = a[removeChunk:]
            # getting data to be deprmotoed 

            a = a[:self.batchSize]
            # updating a 

            b = np.vstack((updateData, b))


        
        return a, b
    
    def updateCache(self, data:np.ndarray) -> np.ndarray:
        """
        Updates the current cach
        
        Keyword arguments:

        data -- Data for the cache to be updated on

        Return: Updated data
        """
        try:
            self.cache1, self.cache2 = self.promoteData(self.cache1, self.cache2)    
        except PrmotionError:
            pass

        try:
            updatedData, self.cache1 = self.promoteData(data, self.cache1)
            return updatedData
        except PrmotionError:
            pass

        return data
        


    def __next__(self):
        """
        Returns the next data
        """
        futureData = self.nextBatch()




        