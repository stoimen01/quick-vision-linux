import cv2
import time
import tensorflow as tf
import numpy as np

from model import ConvNet
from multiprocessing import Process, Queue
from threading import Thread


class NeuralProcess:

    def __init__(self, in_size, tar_size, chkpt):
        self.queue = Queue()
        self.nn_process = Process(target=self.frames_process, args=(self.queue, in_size, tar_size, chkpt,))

    def start_processing(self):
        self.nn_process.start()

    def put_recent_frame(self, frame):
        self.queue.put(frame)

    def frames_process(self, q: Queue, in_size, tar_size, chkpt):

        sess = tf.Session()
        conv_net = ConvNet(sess,
                           in_size,
                           tar_size,
                           chkpt)

        # restoring the exported model
        if not conv_net.load(chkpt):
            print("loading problem occurred")
            return

        print("starting watcher")

        class FramesWatcher:

            def __init__(self, q):
                self.queue = q
                self.stopped = False
                self.isLastFrameProcessed = False
                self.last_frame = None
                self.watcher = Thread(target=self.watch_frames)

            def start(self):
                self.watcher.start()
                print('frames watcher started')

            def watch_frames(self):
                while not self.stopped:

                    # always keeping the last frame
                    self.last_frame = self.queue.get()
                    self.isLastFrameProcessed = False

                    # can be stopped only via the queue
                    if not self.last_frame:
                        self.stopped = True

                print('frames watcher stopped')

        fw = FramesWatcher(q)
        fw.start()

        # starting OpenCV window thread
        window_orig = 'original'
        window_proc = 'processed'
        cv2.startWindowThread()
        cv2.namedWindow(window_orig)
        cv2.namedWindow(window_proc)

        while not fw.stopped:
            if fw.last_frame and not fw.isLastFrameProcessed:

                arr = np.frombuffer(fw.last_frame, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

                t1 = time.time()
                result = conv_net.pass_forward(img)
                print("NN time : " + str(1 / (time.time() - t1)))

                cv2.imshow(window_orig, img)
                cv2.imshow(window_proc, result)

                fw.isLastFrameProcessed = True

    def stop_processing(self):
        # Wait for the worker to finish
        self.queue.put(None)
        self.queue.close()
        self.queue.join_thread()
        self.nn_process.join()
        print('stopping')