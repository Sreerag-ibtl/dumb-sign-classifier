import cv2
# For processing images.
import tensorflow as tf
# For loading model, predicting,.
import numpy as np
# For adding dimension.

class LoadAndPredict:

    def __init__( self ):

        self.label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
                          'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16,
                          'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
                          'Z': 25, 'del': 26, 'nothing': 27, 'space': 28}
    
    def feed_model( self, model_path ):
        """Description: Load the .h5 model to memory.
        Arguments: model_path:- Path to model.
        Returns: """        
        self.model = tf.keras.models.load_model( model_path )
        #Loaded the model.
        
    def preprocess( self, image ):
        """Description: Resize, normalize, expand_dims.
        Arguments: Image array.
        Returns: Preprocessed image."""
        image = cv2.resize( image, ( 256, 256 ) )
        image = image / 255.
        image = np.expand_dims( image, 0 )
        return image

    def predict( self, data ):
        """Description: Obtain prediction from model.
        Arguments: Preprocessed image.
        Returns: Raw prediction."""
        prediction = self.model.predict     ( data )
        cls_index  = self.process_prediction( prediction )
        label      = self.index_to_label( cls_index )
        return label

    def process_prediction( self, prediction ):
        """Description: Process Raw prediction and obtain class.
        Arguments: Raw prediction.
        Returns: Class."""
        prediction = prediction[ 0 ]
        cls_index  = np.argmax ( prediction )
        return cls_index

    def index_to_label( self, cls_index ):
        """Description: Fetch label for the specified index.
        Arguments: Index.
        Returns: Label."""
        keys  = list( self.label_map.keys() )
        label = keys[ cls_index ]
        return label
        
