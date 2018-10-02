import os
import cv2
import time
import json
import datetime
from Service import Service



class Bars_Tones(Service):
    """Class for the Bars and Tones detection service.
        This service, may also be useful for matching other template like frames."""

    def __init__(self, video, miff=None):
        """Constructor for Bars_Tones"""
        self.sample_ratio = 30
        self.image = cv2.imread(os.path.join(".", "cpb-aacip-507-g44hm5369j_109340.png"))
        super().__init__(video)

    def run_service(self):
        cap = cv2.VideoCapture(self.video)
        counter = 0
        bars_tones = []
        start_frame = True

        start_time = "/"  # something went wrong if this doesnt get set
        while cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            if counter % self.sample_ratio == 0:
                sim = cv2.matchTemplate(self.image, f, cv2.TM_SQDIFF_NORMED)
                if sim[0][0] > .5: ## if it is bars and tones
                    if start_frame:
                        start_time = cap.get(cv2.CAP_PROP_POS_MSEC)
                        start_frame = False
                else: ## if its not bars and tones
                    if not start_frame: ## if the start time has been set
                        end_time = cap.get(cv2.CAP_PROP_POS_MSEC)
                        if end_time == "0:0.0":
                            continue
                        start = datetime.timedelta(milliseconds=start_time)
                        end = datetime.timedelta(milliseconds=end_time)
                        bars_tones.append((start.total_seconds(), end.total_seconds()))
                        start_frame = True

            counter += 1
        return {"bars_tones":bars_tones}
