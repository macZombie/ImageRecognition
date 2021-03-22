#!/usr/bin/python3

import jetson.inference
import jetson.utils
import subprocess
import time
import datetime
import simpleaudio as sa


ringer = "bus"
#ringer = "car"


def showImage(myDisplay):

    myDisplay.RenderOnce(img,width,height)
      
    myDisplay.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
    


def ringTheBell():
    filename = 'bell.wav'
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait until sound has finished playing



myThreshold = 0.750


myFile = open("myLog.csv","w")


myFile.write("dateAndTime,object,confidence")

# load the recognition network
# net = jetson.inference.detectNet("ssd-mobilenet-v2" , threshold=myThreshold )
net = jetson.inference.detectNet("ssd-inception-v2" , threshold=myThreshold )


# this is the USB camera
camera = jetson.utils.gstCamera(1280, 720, "/dev/video0" )


display = jetson.utils.glDisplay()


   
while display.IsOpen():
   
   # myDate = datetime.datetime.now()
    
    img, width, height = camera.CaptureRGBA()

    detections = net.Detect(img,width,height)
    
    showImage(display)
       
    for detection in detections:

        myString = str(   detection );
       
        myList = myString.split()
  
        myNumString = myList[4]
        
        myConfidence = myList[7]
     
        myClass = net.GetClassDesc(  int( myList[4] )  );
        
        


        
        if ( len( myClass )  > 1 ):
            
            #showImage(display)
            
            myTime = str( datetime.datetime.now() )
            
            outputString = myTime + "," + myClass + "," + myConfidence
            
            print(outputString)
            
            myFile.write("\n" + outputString)
            
            
               
            commandString = "espeak -p70 -s 200 \"" + myClass + "\""

            returnedValue = subprocess.call(commandString, shell=True)  # returns the exit code in unix
            
            
            if ( myClass == ringer ):
                ringTheBell()
                #    else:
                #    time.sleep(1)
     
     
     
  
            
myFile.close()



    