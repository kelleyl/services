from Service import Service
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import os
import cv2

class Text_Boxes(Service):
    """The text box service generates directories of images, full frame and bounding boxed
        input: video
        output: to service_output directory
        <filename>
            <frame number>
                <full frame.png>
                <box1.png>
                <box2.png>..."""

    def __init__(self,video):
        """Constructor for Text_Boxes"""
        self.sample_ratio = 30
        self.box_min_conf = .5
        super().__init__(video)

    def run_service(self):
        """taken from pyimagesearch"""

        def decode_predictions(scores, geometry):
            # grab the number of rows and columns from the scores volume, then
            # initialize our set of bounding box rectangles and corresponding
            # confidence scores
            (numRows, numCols) = scores.shape[2:4]
            rects = []
            confidences = []

            # loop over the number of rows
            for y in range(0, numRows):
                # extract the scores (probabilities), followed by the
                # geometrical data used to derive potential bounding box
                # coordinates that surround text
                scoresData = scores[0, 0, y]
                xData0 = geometry[0, 0, y]
                xData1 = geometry[0, 1, y]
                xData2 = geometry[0, 2, y]
                xData3 = geometry[0, 3, y]
                anglesData = geometry[0, 4, y]

                # loop over the number of columns
                for x in range(0, numCols):
                    # if our score does not have sufficient probability,
                    # ignore it
                    if scoresData[x] < self.box_min_conf:
                        continue

                    # compute the offset factor as our resulting feature
                    # maps will be 4x smaller than the input image
                    (offsetX, offsetY) = (x * 4.0, y * 4.0)

                    # extract the rotation angle for the prediction and
                    # then compute the sin and cosine
                    angle = anglesData[x]
                    cos = np.cos(angle)
                    sin = np.sin(angle)

                    # use the geometry volume to derive the width and height
                    # of the bounding box
                    h = xData0[x] + xData2[x]
                    w = xData1[x] + xData3[x]

                    # compute both the starting and ending (x, y)-coordinates
                    # for the text prediction bounding box
                    endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                    endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                    startX = int(endX - w)
                    startY = int(endY - h)

                    # add the bounding box coordinates and probability score
                    # to our respective lists
                    rects.append((startX, startY, endX, endY))
                    confidences.append(scoresData[x])

            # return a tuple of the bounding boxes and associated confidences
            return (rects, confidences)

        # initialize the original frame dimensions, new frame dimensions,
        # and ratio between the dimensions
        (W, H) = (None, None)
        (newW, newH) = (320, 320) # newH and newW must a multiple of 32.
        (rW, rH) = (None, None)

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        net = cv2.dnn.readNet(os.path.join(".","frozen_east_text_detection.pb")) # load the model
        cap = cv2.VideoCapture(self.video)

        counter = 0
        text_boxes = {}
        while cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            if counter % self.sample_ratio == 0:
                # resize the frame, maintaining the aspect ratio todo figure out why were doing this
                f = imutils.resize(f, width=1000)
                orig = f.copy()

                # if our frame dimensions are None, we still need to compute the
                # ratio of old frame dimensions to new frame dimensions
                if W is None or H is None:
                    (H, W) = f.shape[:2]
                    rW = W / float(newW)
                    rH = H / float(newH)

                # resize the frame, this time ignoring aspect ratio
                f = cv2.resize(f, (newW, newH))

                # construct a blob from the frame and then perform a forward pass
                # of the model to obtain the two output layer sets
                blob = cv2.dnn.blobFromImage(f, 1.0, (newW, newH),
                                             (123.68, 116.78, 103.94), swapRB=True, crop=False)
                net.setInput(blob)
                (scores, geometry) = net.forward(layerNames)

                # decode the predictions, then  apply non-maxima suppression to
                # suppress weak, overlapping bounding boxes
                (rects, confidences) = decode_predictions(scores, geometry)
                boxes = non_max_suppression(np.array(rects), probs=confidences)

                # loop over the bounding boxes
                box_list = []
                for (startX, startY, endX, endY) in boxes:
                    # scale the bounding box coordinates based on the respective
                    # ratios
                    startX = int(startX * rW)
                    startY = int(startY * rH)
                    endX = int(endX * rW)
                    endY = int(endY * rH)
                    box_list.append((startX,startY,endX,endY))

                text_boxes[counter] = box_list
            counter += 1

        return text_boxes

