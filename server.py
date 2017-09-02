import socket

from video_pipeline import VideoPipeline


class Server:

    def __init__(self, port):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.v_pipeline = VideoPipeline()
        self.stopped = False

    def start(self):
        self.sock.bind(("", self.port))
        self.v_pipeline.start()
        print("system started !")

        while not self.stopped:
            data, address = self.sock.recvfrom(496)  # buffer size is 496 bytes
            self.v_pipeline.push(data)

        self.v_pipeline.stop()
        self.sock.close()

    def stop(self):
        self.stopped = True
