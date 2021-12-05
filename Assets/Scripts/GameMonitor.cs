using System.IO;
using UnityEngine;

public class GameMonitor : MonoBehaviour {
	public HumanCarController controller;
	public SensorManager rayParser;

	bool recording = false;

	StreamWriter writer;


	public void SetRecording(bool value) {
		recording = value;
		if (recording) {
			writer = File.CreateText("data.csv");
			string titleEntry = "input;";
			foreach (string title in rayParser.titles) {
				titleEntry += title + ";";
			}
			titleEntry = titleEntry.TrimEnd(';');
			writer.WriteLine(titleEntry);
		} else {
			if (writer != null) {
				writer.Close();
				writer = null;
			}
		}
	}

	void Update() {
		if (recording) {
			string entry = "";
			entry += controller.steerValue + ";";
			foreach (float value in rayParser.rayValues) {
				entry += value.ToString() + ";";
			}
			entry = entry.TrimEnd(';');
			writer.WriteLine(entry);
		}
	}
}
