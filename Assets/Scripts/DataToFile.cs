using System.IO;
using UnityEngine;

public class DataToFile : MonoBehaviour
{
	public CarController carController;
	public RayParser rayParser;

	bool recording = false;

	StreamWriter writer;


	void OnGUI()
	{
		GUILayout.BeginArea(new Rect(Screen.width - 120, Screen.height - 70, 100, 50));

		if (recording && GUILayout.Button("End recording"))
		{
			recording = false;
			if (writer != null)
			{
				writer.Close();
				writer = null;
			}
		}
		else if (!recording && GUILayout.Button("Start recording"))
		{
			recording = true;
			writer = File.CreateText("data.csv");
			string titleEntry = "input;";
			foreach (string title in rayParser.titles)
			{
				titleEntry += title + ";";
			}
			titleEntry = titleEntry.TrimEnd(';');
			writer.WriteLine(titleEntry);
		}
		GUILayout.EndArea();
	}

	void Update()
	{
		if (recording)
		{
			string entry = "";
			entry += carController.input + ";";
			foreach (float value in rayParser.rayValues)
			{
				entry += value.ToString() + ";";
			}
			entry = entry.TrimEnd(';');
			writer.WriteLine(entry);
		}
	}
}
