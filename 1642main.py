import serial
import time
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import random as rand

# Change the configuration file name
configFileName = 'Config Files/1642.cfg'

CLIport = {}
Dataport = {}
byteBuffer = np.zeros(2**15, dtype = 'uint8')
byteBufferLength = 0
tracked_objects = {} # Store the information of tracked objects
entries = 0 
exits = 0

BOUNDARY_X = -0.5
BOUNDARY_Y = 1
BOUNDARY_WIDTH = 1
BOUNDARY_HEIGHT = 1

# ------------------------------------------------------------------

# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):
    
    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports
    
    # Raspberry pi
    #CLIport = serial.Serial('/dev/ttyACM0', 115200)
    #Dataport = serial.Serial('/dev/ttyACM1', 921600)
    
    # Windows
    # CLIport = serial.Serial('COM8', 115200)
    # Dataport = serial.Serial('COM9', 921600)
    
    # MacOS - AWR1642
    # CLIport = serial.Serial('/dev/tty.usbmodemR00410271', 115200)
    # Dataport = serial.Serial('/dev/tty.usbmodemR00410274', 921600)
    
    # MacOS - AWR1642
    CLIport = serial.Serial('/dev/tty.usbmodemR00410271', 115200)
    Dataport = serial.Serial('/dev/tty.usbmodemR00410274', 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i+'\n').encode())
        print(i)
        time.sleep(0.01)
        
    return CLIport, Dataport

# ------------------------------------------------------------------

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {} # Initialize an empty dictionary to store the configuration parameters
    
    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        
        # Split the line
        splitWords = i.split(" ")
        
        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 3
        
        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1
            
            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2
                
            digOutSampleRate = int(splitWords[11])
            
        # Get the information about the frame configuration    
        elif "frameCfg" in splitWords[0]:
            
            chirpStartIdx = int(splitWords[1])
            chirpEndIdx = int(splitWords[2])
            numLoops = int(splitWords[3])
            numFrames = int(splitWords[4])
            framePeriodicity = float(splitWords[5])

            
    # Combine the read data to obtain the configuration parameters           
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
    
    return configParameters
   
# ------------------------------------------------------------------

# Funtion to read and parse the incoming data
def readAndParseData18xx(Dataport, configParameters):
    global byteBuffer, byteBufferLength
    
    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12
    BYTE_VEC_ACC_MAX_SIZE = 2**15
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1
    MMWDEMO_UART_MSG_RANGE_PROFILE   = 2
    maxBufferSize = 2**15
    tlvHeaderLengthInBytes = 8
    pointLengthInBytes = 16
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
    
    # Initialize variables
    magicOK = 0 # Checks if magic number has been read
    dataOK = 0 # Checks if the data has been read correctly
    frameNumber = 0
    detObj = {}
    
    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype = 'uint8')
    byteCount = len(byteVec)
    
    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount
        
    # Check that the buffer has some data
    if byteBufferLength > 16:
        
        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc+8]
            if np.all(check == magicWord):
                startIdx.append(loc)
               
        # Check that startIdx is not empty
        if startIdx:
            
            # Remove the data before the first start index
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength-startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]),dtype = 'uint8')
                byteBufferLength = byteBufferLength - startIdx[0]
                
            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0
                
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]
            
            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12+4],word)
            
            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1
    
    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2**8, 2**16, 2**24]
        
        # Initialize the pointer index
        idX = 0
        
        # Read the header
        magicNumber = byteBuffer[idX:idX+8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX:idX+4],word),'x')
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        platform = format(np.matmul(byteBuffer[idX:idX+4],word),'x')
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        subFrameNumber = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4

        # Read the TLV messages
        for tlvIdx in range(numTLVs):
            
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]

            # Check the header of the TLV message
            tlv_type = np.matmul(byteBuffer[idX:idX+4],word)
            idX += 4
            tlv_length = np.matmul(byteBuffer[idX:idX+4],word)
            idX += 4

            # Read the data depending on the TLV message
            if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:

                # Initialize the arrays
                x = np.zeros(numDetectedObj,dtype=np.float32)
                y = np.zeros(numDetectedObj,dtype=np.float32)
                z = np.zeros(numDetectedObj,dtype=np.float32)
                velocity = np.zeros(numDetectedObj,dtype=np.float32)
                
                for objectNum in range(numDetectedObj):
                    
                    # Read the data for each object
                    x[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)[0]
                    idX += 4
                    y[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)[0]
                    idX += 4
                    z[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)[0]
                    idX += 4
                    velocity[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)[0]
                    idX += 4
                
                # Store the data in the detObj dictionary
                detObj = {"numObj": numDetectedObj, "x": x, "y": y, "z": z, "velocity": velocity}
                dataOK = 1
                
 
        # Remove already processed data
        if idX > 0 and byteBufferLength>idX:
            shiftSize = totalPacketLen
            
                
            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),dtype = 'uint8')
            byteBufferLength = byteBufferLength - shiftSize
            
            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0         

    return dataOK, frameNumber, detObj

# ------------------------------------------------------------------

# Funtion to update the data and display in the plot
def update(configParameters, p):
     
    dataOk = 0
    global detObj
    x = []
    y = []
      
    # Read and parse the received data
    dataOk, frameNumber, detObj = readAndParseData18xx(Dataport, configParameters)
    
    if dataOk and len(detObj["x"]) > 0:
        
        x = -detObj["x"]
        y = detObj["y"]
        
        ### Perform DBSCAN clustering
        X = np.column_stack((x, y))
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        labels = dbscan.fit_predict(X) # [0, -1, 1, 0, 2, 0, -1, 1...] 0th, 3rd, and 5th element belong to 0th cluster etc...
        
        # Plot data points coloured by DBSCAN cluster labels
        for label in np.unique(labels):
            if label == -1:  # Outliers
                mask = (labels == label) # e.g. mask = [False, True, False, False, False, False, True, False...]
                p.plot(x[mask], y[mask], pen=None, symbol='o', symbolBrush=(255, 0, 0), symbolSize=5)
            else:  # Inliers
                mask = (labels == label) # In the inliers case, a label is a single cluster
                p.plot(x[mask], y[mask], pen=None, symbol='o', symbolBrush=(1, 50, 32), symbolSize=5)
                points = np.column_stack((x[mask], y[mask]))
                
                # Perform Convex Hull and get the 'area' of object FIXME - Currently no point doing convex hull when average of data returned from bird is 2 points
                '''
                # hull = ConvexHull(points)
                # for simplex in hull.simplices:
                #     p.plot(points[simplex, 0], points[simplex, 1], pen=pg.mkPen(color='k', width=3))
                # text = pg.TextItem(str(format(hull.area, '.3f'))) # Show the hull area
                # p.addItem(text)
                # text.setPos(points[simplex, 0][0], points[simplex, 1][0])
                '''
                
                ### OBJECT TRACKING
                # Centre location of cluster
                centroid = np.mean(X[mask], axis = 0)

                found_match = False
                for object_id, object_info in tracked_objects.items(): # For each, DBSCAN label, the last object in tracked_objects which is in range will be the tracked on.
                    if np.linalg.norm(centroid - object_info["position"]) < 0.3: # Calculate the euclidian distance
                        # check_entries_exits(centroid[0], centroid[1], object_info["position"][0], object_info["position"][1]) # FIXME - Need to move this
                        tracked_objects[object_id]["prev_position"] = tracked_objects[object_id]["position"]
                        tracked_objects[object_id]["position"] = centroid
                        tracked_objects[object_id]["frame_count"] += 1
                        found_match = True
                    
                if not found_match:
                    tracked_objects[len(tracked_objects)] = {"prev_position": centroid, "position": centroid, "frame_count": frameNumber, "colour": None} # Add a new object with the current frame number
                    
        # Remove old objects if it doesn't show up in frames after 5 seconds: 150 S / 30 FPS
        lost_objects = [object_id for object_id, object_info in tracked_objects.items() if (frameNumber - object_info["frame_count"]) > 150]
        for object_id in lost_objects:
            tracked_objects.pop(object_id)
            
        # Check previous and current positions of all tracked objects to see if any have crossed the boundary.
        for _, object_info in tracked_objects.items():
            check_entries_exits(object_info["prev_position"][0], object_info["prev_position"][1], object_info["position"][0], object_info["position"][1])
            
        # VISUALISE TRACKS
        visualise_tracked_objects(p)
                
        # Add circular grid lines
        create_circular_grid(p)
        
        # Add boundaries for counting
        create_boundaries(p)
        
        # Create a displate number of entries and exits
        create_entries_exits_text(p)
        
        # s.setData(x, y) # FIXME - What does this even do?
        QtWidgets.QApplication.processEvents()
        
        # Clear previous scatter plot items so plots from next frame(s) don't overlap
        p.clear()
    
    return dataOk

# -----------------------------------------------------------------

def create_entries_exits_text(p):
    text_counts = pg.TextItem("Entries:" + str(entries) + "\n Exits:" + str(exits))
    p.addItem(text_counts)
    text_counts.setPos(-2, 3)

# -----------------------------------------------------------------

def check_entries_exits(xi, yi, x, y):
    global entries, exits
    # Check if previous position was inside and current position is outside
    if (xi > BOUNDARY_X and xi < (BOUNDARY_X + BOUNDARY_WIDTH)) and (yi > BOUNDARY_Y and yi < (BOUNDARY_Y + BOUNDARY_HEIGHT)):
        if (x < BOUNDARY_X or x > (BOUNDARY_X + BOUNDARY_WIDTH)) or (y < BOUNDARY_Y or y > (BOUNDARY_Y + BOUNDARY_HEIGHT)):
            exits += 1
    # Check if previous position was outside and current position is inside
    if (x > BOUNDARY_X and x < (BOUNDARY_X + BOUNDARY_WIDTH)) and (y > BOUNDARY_Y and y < (BOUNDARY_Y + BOUNDARY_HEIGHT)):
        if (xi < BOUNDARY_X or xi > (BOUNDARY_X + BOUNDARY_WIDTH)) and (yi < BOUNDARY_Y or yi > (BOUNDARY_Y + BOUNDARY_HEIGHT)):
            entries += 1


# -----------------------------------------------------------------

def visualise_tracked_objects(p):
    

    for _, object_info in tracked_objects.items():
        if object_info["colour"] is None:
            red = rand.randint(0, 255)
            blue = rand.randint(0, 255)
            green = rand.randint(0, 255)
            object_info["colour"] = (red, blue, green) # Keep the same colour even when other objects are deleted from tracked_objects - Stop the shifting of colours.
        centroid = object_info["position"]
        p.plot([centroid[0]], [centroid[1]], pen=None, symbol='x', symbolBrush=object_info["colour"], symbolSize=20)  

# -----------------------------------------------------------------

def create_circular_grid(p):
    num_lines = 5
    radius = 1
    angles = np.linspace(0, 2*np.pi, 100)
    for i in range(1, num_lines + 1):
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        p.plot(x, y, pen=pg.mkPen(color='g', width=2, style=pg.QtCore.Qt.DotLine))
        radius += 1
        
# -----------------------------------------------------------------

def create_boundaries(p):
    # Define rectangle coordinates and size
    rect = QtWidgets.QGraphicsRectItem(BOUNDARY_X, BOUNDARY_Y, BOUNDARY_WIDTH, BOUNDARY_HEIGHT)  # (x, y, width, height)
    pen = pg.mkPen(color='r', width=2)
    rect.setPen(pen)
    p.addItem(rect)
    
    # Add radial boundaries - Azmiuth 30 deg        
    p.plot(x = [-5.77, 0, 5.77], y = [10, 0, 10], pen=pg.mkPen(color='g', width=2))

# -------------------------    MAIN   -----------------------------

def main():
    
    # Configurate the serial port
    CLIport, Dataport = serialConfig(configFileName)

    # Get the configuration parameters from the configuration file
    configParameters = parseConfigFile(configFileName)
    
    # START QtAPPfor the plot
    app = QtWidgets.QApplication([])

    # Set the plot 
    pg.setConfigOption('background', 'b')
    win = pg.GraphicsLayoutWidget(title="2D scatter plot")
    p = win.addPlot()
    p.setXRange(-1.5, 1.5)
    p.setYRange(0, 3)
    p.setLabel('left',text = 'Y position (m)')
    p.setLabel('bottom', text= 'X position (m)')
    p.setLabel('right',text =None)
    p.setLabel('top', text=None)
    # p.showGrid(True, True)
    # s = p.plot([],[],pen=None,symbol='o')

    # Set the border color and width
    border_pen = pg.mkPen(color='g', width=2)
    p.getAxis('left').setPen(border_pen)  # Left border
    p.getAxis('bottom').setPen(border_pen)  # Bottom border
    p.getAxis('right').setPen(border_pen)  # Right border
    p.getAxis('top').setPen(border_pen)  # Top border

    p.setAspectLocked()    

    win.show()
    
    rand.seed(a="'Colour' is the right way to spell it in Australia") # Lets create a deterministic random number generator - Given the same seed.

    # Main loop 
    detObj = {}
    frameData = {}
    currentIndex = 0
    while True:
        try:
            # Update the data and check if the data is okay
            dataOk = update(configParameters, p)
            
            if dataOk:
                # Store the current frame into frameData
                frameData[currentIndex] = detObj
                currentIndex += 1
            
            # time.sleep(0.01) # Sampling frequency of 30 Hz ### \\TODO - HINT INTO FRAME RATE ISSUE???
            
        # Stop the program and close everything if Ctrl + c is pressed
        except KeyboardInterrupt:
            CLIport.write(('sensorStop\n').encode())
            CLIport.close()
            Dataport.close()
            win.close()
            break
        
if __name__ == "__main__":
    main()