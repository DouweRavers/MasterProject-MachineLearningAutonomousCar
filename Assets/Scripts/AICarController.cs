using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

[RequireComponent(typeof(CarBase))]
public class AICarController : MonoBehaviour {
	SensorManager sensor;
	CarBase car;
	static Thread pythonThread;
	static TcpClient client;
	static TcpListener listener;
	bool pythonCommunication = false;
	float[] visionSensorValues;
	float calculatedSteer = 0f;

	void Start() {
		if (UI.player != 1) {
			Destroy(gameObject);
			return;
		}
		car = GetComponent<CarBase>();
		sensor = GetComponentInChildren<SensorManager>();
		ConnectToPython();
	}
	bool toggler = false;
	void Update() {
		if (!pythonCommunication) {
			toggler = true;
			return;
		} else if (toggler) {
			car.toggleEngine();
			toggler = false;
		}
		visionSensorValues = sensor.rayValues;
		car.steerCar(calculatedSteer);
	}

	public void ConnectToPython() {
		if (pythonThread != null) pythonThread.Abort();
		ThreadStart ts = new ThreadStart(StartThread);
		pythonThread = new Thread(ts);
		pythonThread.Start();
	}

	void StartThread() {
		if (client != null) client.Close();
		if (listener != null) listener.Stop();
		client = null;
		listener = null;
		IPAddress localAdd = IPAddress.Parse("127.0.0.1");
		listener = new TcpListener(localAdd, 25001);
		listener.Start();
		print("start listener");
		client = listener.AcceptTcpClient();
		print("found client");


		pythonCommunication = true;
		while (pythonCommunication) {
			if (!client.Connected) break;
			if (SendCurrentVision(client)) continue;
			if (ReceiveSteering(client)) continue;
		}
		listener.Stop();
	}

	bool SendCurrentVision(TcpClient client) {
		NetworkStream networkStream = client.GetStream();
		string message = "{";
		if (visionSensorValues == null || visionSensorValues.Length == 0) return true;
		foreach (float value in visionSensorValues) {
			message += value + ";";
		}
		message = message.TrimEnd(';') + "}";
		byte[] messageBuffer = Encoding.UTF8.GetBytes(message);
		networkStream.Write(messageBuffer, 0, messageBuffer.Length);
		return false;
	}
	bool ReceiveSteering(TcpClient client) {
		NetworkStream networkStream = client.GetStream();
		byte[] messageBuffer = new byte[client.ReceiveBufferSize];

		int messageBufferSize = networkStream.Read(messageBuffer, 0, client.ReceiveBufferSize);
		string receivedMessage = Encoding.UTF8.GetString(messageBuffer, 0, messageBufferSize);
		if (receivedMessage != null || receivedMessage.Length != 0) {
			calculatedSteer = -1 * float.Parse(receivedMessage.Replace('.', ','));
			print(calculatedSteer);
			return false;
		} else return true;
	}
}