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
	Thread pythonThread;
	bool pythonCommunication = false;
	float[] visionSensorValues;
	float calculatedSteer = 0f;

	void Start() {
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

	void Destroy() {
		DisconnectPython();
	}

	public void ConnectToPython() {
		ThreadStart ts = new ThreadStart(StartThread);
		pythonThread = new Thread(ts);
		pythonThread.Start();
	}

	public void DisconnectPython() {
		pythonCommunication = false;
	}

	void StartThread() {
		IPAddress localAdd = IPAddress.Parse("127.0.0.1");
		TcpListener listener = new TcpListener(IPAddress.Any, 25001);
		listener.Start();
		TcpClient client = listener.AcceptTcpClient();
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

		if (receivedMessage != null) {
			calculatedSteer = float.Parse(receivedMessage.Trim('{', '}').Replace('.', ','));
			return false;
		} else return true;
	}
}
