import zlib
import time
from nn_process import NeuralProcess


class Packet:

    def __init__(self, data):
        self.time_stamp = int.from_bytes(data[0:8], byteorder='big')
        self.packet_number = int.from_bytes(data[8:12], byteorder='big')
        self.packet_sum = int.from_bytes(data[12:16], byteorder='big')
        self.img_data = data[16:497]


class VideoPipeline:

    def __init__(self, in_size, tar_size, chkpt):
        self.packets_dic = {}  # timestamps for keys and array of packets for values
        self.recent_ts = 0  # timestamp from last received frame
        self.nn = NeuralProcess(in_size, tar_size, chkpt)

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