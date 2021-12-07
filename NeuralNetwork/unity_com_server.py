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
        # received_data = "{1; 0, 635382; 1; 0, 4389649; 0, 9414662; 0, 3333805; 0, 4457624; 0, 2698263; 0, 3131007; 0, 2292469; 0, 2516059; 0, 2021143; 0, 2168587; 0, 1844587; 0, 1959085; 0, 1745058; 0, 1822734; 0, 1696527; 0, 1741297; 0, 1705346; 0, 1700388}"
        received_data = received_data.strip("{")
        received_data = received_data.strip("}")
        received_data = received_data.replace(",", ".")
        received_data = received_data.replace(" ", "")
        string_data = received_data.split(";")

        float_data = [float(var) for var in string_data]
        return float_data

    def send_steering_value(self, value):
        self.sock.sendall(str(value).encode("UTF-8"))
