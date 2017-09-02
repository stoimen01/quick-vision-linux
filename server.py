import socket

from video_pipeline import VideoPipeline


class Server:

    def __init__(self, port, in_size, tar_size, chkpt):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.v_pipeline = VideoPipeline(in_size, tar_size, chkpt)
        self.stopped = False

    def start(self):
        self.sock.bind(("", self.port))
        self.v_pipeline.start()
        print("waiting video provider !")

        while not self.stopped:
            data, address = self.sock.recvfrom(496)  # buffer size is 496 bytes
            self.v_pipeline.push(data)

        self.v_pipeline.stop()
        self.sock.close()

    def stop(self):
        self.stopped = True
