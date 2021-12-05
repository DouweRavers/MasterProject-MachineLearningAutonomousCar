using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Sensors;
public class SensorManager : MonoBehaviour {
	public RayPerceptionSensorComponent3D rayPerceptionSensorComponent3D;
	public float[] rayValues {
		get {
			RayPerceptionOutput output = RayPerceptionSensor.Perceive(rayPerceptionSensorComponent3D.GetRayPerceptionInput());
			float[] values = new float[output.RayOutputs.Length];
			for (int i = 0; i < output.RayOutputs.Length; i++) {
				values[i] = output.RayOutputs[i].HitFraction;
			}
			return values;
		}
	}

	public string[] titles {
		get {
			RayPerceptionInput rayPerceptionInput = rayPerceptionSensorComponent3D.GetRayPerceptionInput();
			string[] values = new string[rayPerceptionInput.Angles.Count];
			for (int i = 0; i < rayPerceptionInput.Angles.Count; i++) {
				values[i] = "Ray at angle: " + rayPerceptionInput.Angles[i];
			}
			return values;
		}
	}

	public float[] angles {
		get {
			RayPerceptionInput rayPerceptionInput = rayPerceptionSensorComponent3D.GetRayPerceptionInput();
			float[] values = new float[rayPerceptionInput.Angles.Count];
			for (int i = 0; i < rayPerceptionInput.Angles.Count; i++) {
				values[i] = rayPerceptionInput.Angles[i];
			}
			return values;
		}
	}

	void Start() {
		rayPerceptionSensorComponent3D = GetComponent<RayPerceptionSensorComponent3D>();
	}

	// void OnGUI()
	// {
	// 	for (int i = 0; i < rayValues.Length; i++)
	// 	{
	// 		GUILayout.BeginHorizontal();
	// 		GUILayout.Label(titles[i]);
	// 		GUILayout.HorizontalSlider(rayValues[i], 0f, 1f, GUILayout.Width(200));
	// 		GUILayout.EndHorizontal();
	// 	}
	// }
}
