import socket
import datetime

listeningAddress = ("192.168.0.28", 7070)

datagramSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
datagramSocket.bind(listeningAddress)

while(True):
    tempVal, sourceAddress = datagramSocket.recvfrom(128)
    #print(int.from_bytes(tempVal, "big"))
    print(tempVal.decode())