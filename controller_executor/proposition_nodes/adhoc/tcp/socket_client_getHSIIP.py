#!/usr/bin/env python

import socket
import sys
import os

class TCP:
	def __init__(self):
		# Create a TCP/IP socket
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

		# Connect the socket to the port on the server given by the caller
		if(len(sys.argv)>1):
			server_address = (sys.argv[1], 65432)
		else:
			server_address = ("132.236.59.220", 65432)
		print >>sys.stderr, 'connecting to %s port %s' % server_address
		self.sock.connect(server_address)

		self.ip = os.popen('ip addr show wifi0').read().split("inet ")[1].split("/")[0]
		#self.ip = os.popen('ip addr show wlan0').read().split("inet ")[1].split("/")[0]
		self.hsiip = "127.0.0.1"

	def getHSIIP(self):
		try:		    
		    print >>sys.stderr, 'sending "%s"' % self.ip
		    self.sock.sendall("hsiip")

		    amount_received = 0
		    amount_expected = len(self.ip)
		    while amount_received < amount_expected:
			recv_data = self.sock.recv(32)
			amount_received += len(recv_data)
			print >>sys.stderr, 'received "%s"' % recv_data
			self.hsiip = recv_data

		finally:
			pass 
	
	def sendData(self, data):
		try:		    
		    print >>sys.stderr, 'sending "%s"' % data
		    self.sock.sendall(data)

		    amount_received = 0
		    amount_expected = len(data)
		    while amount_received < amount_expected:
			recv_data = self.sock.recv(len(data))
			amount_received += len(recv_data)
			print >>sys.stderr, 'received "%s"' % recv_data
		
		finally:
			pass

	def closeSocket(self):
		self.sock.close()

if __name__ == "__main__":
	tcp = TCP()
	#tcp.getHSIIP()
	tcp.sendData("Petersen")
	tcp.closeSocket()
	
