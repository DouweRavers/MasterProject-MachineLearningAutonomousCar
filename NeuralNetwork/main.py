import socket
import time
import csv

host, port = "127.0.0.1", 25001
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))

with open('data.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile, delimiter=";")
    for row in csvReader:
        if row[0] == "input":
            continue
        time.sleep(0.016)
        receivedData = sock.recv(1024).decode("UTF-8")
        value = "{" + row[0] + "}"
        sock.sendall(value.encode("UTF-8"))
