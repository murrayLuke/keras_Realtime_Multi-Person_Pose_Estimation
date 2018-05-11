#!/usr/bin/env python

from threading import Thread, Lock
import cv2
import pylab as plt
from queue import Queue
import time

class WebcamVideoStream :
    def __init__(self, src = 0) :
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

if __name__ == "__main__" :
    vs = WebcamVideoStream().start()
    while True :
        frame = vs.read()
        cv2.imshow('webcam', frame)
        key = cv2.waitKey(1)
        if key == 27: # esc to quit
            break 
        if key == 32: # space to use image
            cv2.imshow('results', frame)
            # visualizer.write(frame)
            # plt.imshow(frame[:, :, [2, 1, 0]])
            # plt.show()

    vs.stop()
    cv2.destroyAllWindows()