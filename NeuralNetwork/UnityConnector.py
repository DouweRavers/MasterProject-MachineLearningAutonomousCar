import socket


class UnityConnector:
    def __init__(self):
        self.host = "127.0.0.1"
        self.port = 25001
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("start connecting")
        self.sock.connect((self.host, self.port))

    def request_sensor_values(self):
        received_data = self.sock.recv(1024).decode("UTF-8")
        received_data = received_data.strip("{")
        received_data = received_data.strip("}")
        received_data = received_data.replace(",", ".")
        received_data = received_data.replace(" ", "")
        string_data = received_data.split(";")
        float_data = [float(var) for var in string_data]
        return float_data

    def send_steering_value(self, value):
        self.sock.sendall(str(value).encode("UTF-8"))
