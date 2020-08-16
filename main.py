from classifier import LoadAndPredict
#Implementation of load and predict.
import argparse
#To read terminal arguments.
import cv2
#Video stream.
import pyttsx3
#For audio output.

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser . add_argument( "--model",  default="model/weights.05-1.49.hdf5" )
    parser . add_argument( "--camera", default=0 )

    args   = parser.parse_args( )

    engine = pyttsx3.init()
    
    lp = LoadAndPredict( )
    lp . feed_model( args.model )
    print( "[INFO]Fed the model succesfully." )

    x1, y1, x2, y2 = ( 100, 100, 300, 300 )

    #Open a camera.
    camera = cv2.VideoCapture( args.camera )

    #Start an indefinite video loop.
    while True:
        #Read a frame.
        flag, frame = camera.read( )
        #If no error in reading.
        if flag:
            #Preprocess the image.
            frame      = cv2.flip( frame, 1 )
            display    = frame.copy( )
            frame      = frame[ y1:y2, x1:x2 ]
            data       = lp.preprocess( frame )
            prediction = lp.predict( data )
            engine.say( prediction )
            engine.runAndWait()
            print( "[INFO]Prediction:{}".format( prediction ) )
            cv2.rectangle( display, (x1, y1), (x2, y2), (0, 0, 255), 4 )
            cv2.imshow( "Frame", display )
            cv2.imshow( "Crop",  frame )
            #If q is presses stop stream.
            if cv2.waitKey( 1 ) == ord( "q" ):
                break
        else:
            #If error in reading.
            break
    #Cleaning.
    camera.release()
    cv2.destroyAllWindows()
