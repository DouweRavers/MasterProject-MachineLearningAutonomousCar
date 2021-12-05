using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class DataDisplay : MonoBehaviour {
	public CarBase car;
	public SensorManager sensors;
	public Slider left, right, root;

	Slider[] sensorDisplay;
	void Start() {
		float[] angles = sensors.angles;
		sensorDisplay = new Slider[angles.Length];
		root.transform.eulerAngles = Vector3.forward * angles[0];
		sensorDisplay[0] = root;
		for (int i = 1; i < angles.Length; i++) {
			GameObject sliderObject = Instantiate(root.gameObject);
			sliderObject.transform.SetParent(root.transform.parent);
			sliderObject.transform.position = root.transform.position;
			sliderObject.transform.eulerAngles = Vector3.forward * angles[i];
			sensorDisplay[i] = sliderObject.GetComponent<Slider>();
		}
	}

	void Update() {
		left.value = -car.steerValue;
		right.value = car.steerValue;
		float[] values = sensors.rayValues;
		for (int i = 0; i < sensorDisplay.Length; i++) {
			sensorDisplay[i].value = values[i];
		}
	}
}
