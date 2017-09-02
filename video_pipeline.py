import zlib
import numpy as np
import tensorflow as tf
import cv2
import json
import base64
import time
from multiprocessing import Process, Queue
from threading import Thread


class Packet:

    def __init__(self, data):
        self.time_stamp = int.from_bytes(data[0:8], byteorder='big')
        self.packet_number = int.from_bytes(data[8:12], byteorder='big')
        self.packet_sum = int.from_bytes(data[12:16], byteorder='big')
        self.img_data = data[16:497]


class NeuralNetwork:

    def __init__(self):
        self.queue = Queue()
        self.nn_process = Process(target=self.frames_process, args=(self.queue,))

    def start_processing(self):
        self.nn_process.start()

    def put_recent_frame(self, frame):
        self.queue.put(frame)

    def frames_process(self, q: Queue):

        isLastFrameProcessed = False

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
                    if not self.last_frame:
                        self.stopped = True
                print('frames watcher stopped')

        fw = FramesWatcher(q)
        fw.start()

        # restoring the exported model
        model_dir = 'exported_model/'
        sess = tf.Session()
        saver = tf.train.import_meta_graph(model_dir + "export.meta")
        saver.restore(sess, model_dir + "export")
        input_vars = json.loads(tf.get_collection("inputs")[0].decode('utf-8'))
        output_vars = json.loads(tf.get_collection("outputs")[0].decode('utf-8'))
        input = tf.get_default_graph().get_tensor_by_name(input_vars["input"])
        output = tf.get_default_graph().get_tensor_by_name(output_vars["output"])
        print("model ready!")

        # starting OpenCV window thread
        window_title = 'preview'
        cv2.startWindowThread()
        cv2.namedWindow(window_title)

        while not fw.stopped:
            if fw.last_frame and not fw.isLastFrameProcessed:
                # always processing the last frame
                t1 = time.time()

                # processing with the neural network
                input_instance = dict(input=base64.urlsafe_b64encode(fw.last_frame).decode("ascii"), key="0")
                input_instance = json.loads(json.dumps(input_instance))
                input_value = np.array(input_instance["input"])

                output_value = sess.run(output, feed_dict={input: np.expand_dims(input_value, axis=0)})[0]
                output_instance = dict(output=output_value.decode("ascii"), key="0")
                b64data = output_instance["output"]
                b64data += "=" * (-len(b64data) % 4)
                output_data = base64.urlsafe_b64decode(b64data.encode("ascii"))

                # converting output to OpenCV image
                nparr = np.fromstring(output_data, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                cv2.imshow(window_title, img_np)

                fw.isLastFrameProcessed = True

                print("FPS : " + str(1 / (time.time() - t1)))

    def stop_processing(self):
        # Wait for the worker to finish
        self.queue.put(None)
        self.queue.close()
        self.queue.join_thread()
        self.nn_process.join()
        print('stopping')


class VideoPipeline:

    def __init__(self):
        self.packets_dic = {}  # timestamps for keys and array of packets for values
        self.recent_ts = 0  # timestamp from last received frame
        self.nn = NeuralNetwork()

    def start(self):
        self.nn.start_processing()

    def push(self, data):

        # creating new packet from the received data
        packet = Packet(data)

        # dropping the packet if it is from older frame
        if packet.time_stamp < self.recent_ts:
            del packet
            return

        # searching for existing packets array with the same timestamp as key
        if packet.time_stamp in self.packets_dic:

            # appending the new packet
            self.packets_dic[packet.time_stamp].append(packet)

            # creating new recent frame if the data is complete
            if len(self.packets_dic[packet.time_stamp]) == packet.packet_sum:

                # building the frame from the packets array
                s_packets = sorted(self.packets_dic[packet.time_stamp], key=lambda x: x.packet_number)
                img_data = bytearray()
                i = 0
                while i < len(s_packets):
                    img_data.extend(s_packets[i].img_data)
                    i += 1

                # sending the data to the nn process
                self.nn.put_recent_frame(zlib.decompress(img_data))

                # updating and cleaning
                self.recent_ts = packet.time_stamp
                self.clean_old_packets()
        else:
            # creating new key value pair
            self.packets_dic[packet.time_stamp] = [packet]

    def clean_old_packets(self):

        # cleaning everything older or equal to the most recent timestamp
        for key_ts in dict(self.packets_dic):
            if key_ts <= self.recent_ts:
                del self.packets_dic[key_ts]

    def stop(self):
        self.nn.stop_processing()