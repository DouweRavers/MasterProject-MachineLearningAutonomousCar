using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class CarBase : MonoBehaviour {
	public float brakeTorque = 1000000, motorTorque = 5000, steerAngle = 15;


	public WheelCollider wheelLF, wheelRF, wheelLB, wheelRB;
	public Transform wheelVisualLF, wheelVisualRF, wheelVisualLB, wheelVisualRB;
	public AudioSource startMotor, idle, gasLowRPM, engineOut;
	public float steerValue { get { return steering; } }
	Rigidbody body;
	float steering;
	float maxSpeed = 0, minSpeed = 0;

	public void toggleEngine() {
		if (maxSpeed > 0) maxSpeed = 0;
		else maxSpeed = 12.5f;
		if (minSpeed > 0) {
			minSpeed = 0;
			engineOut.Play();
		} else {
			minSpeed = 10;
			startMotor.Play();
		}
	}

	public void steerCar(float value) {
		steering = Mathf.Clamp(value, -1, 1);
	}

	void Start() {
		body = GetComponent<Rigidbody>();
	}

	void Update() {
		UpdateWheelVisuals();
		UpdateSounds();
	}

	void FixedUpdate() {
		UpdateSpeed();
		UpdateSteering();
	}

	void UpdateWheelVisuals() {
		Vector3 position;
		Quaternion rotation;
		wheelLF.GetWorldPose(out position, out rotation);
		wheelVisualLF.position = position + wheelVisualLF.right * 0.2f;
		wheelVisualLF.rotation = rotation;

		wheelLB.GetWorldPose(out position, out rotation);
		wheelVisualLB.position = position + wheelVisualLF.right * 0.2f;
		wheelVisualLB.rotation = rotation;

		wheelRF.GetWorldPose(out position, out rotation);
		wheelVisualRF.position = position + wheelVisualLF.right * -0.2f;
		wheelVisualRF.rotation = rotation;

		wheelRB.GetWorldPose(out position, out rotation);
		wheelVisualRB.position = position + wheelVisualLF.right * -0.2f;
		wheelVisualRB.rotation = rotation;
	}

	void UpdateSteering() {
		// steering
		if (steering != 0) {
			wheelLF.steerAngle = steerAngle * steering;
			wheelRF.steerAngle = steerAngle * steering;
			steering = 0;
		} else {
			wheelLF.steerAngle = 0;
			wheelRF.steerAngle = 0;
		}
	}

	void UpdateSpeed() {
		if (startMotor.isPlaying) return;
		// Brakes
		if (body.velocity.magnitude > maxSpeed) {
			wheelLF.brakeTorque = brakeTorque;
			wheelRF.brakeTorque = brakeTorque;
			wheelLB.brakeTorque = brakeTorque;
			wheelRB.brakeTorque = brakeTorque;
		} else {
			wheelLF.brakeTorque = 0;
			wheelRF.brakeTorque = 0;
			wheelLB.brakeTorque = 0;
			wheelRB.brakeTorque = 0;
		}

		// Motor
		if (body.velocity.magnitude < minSpeed) {
			wheelLF.motorTorque = motorTorque;
			wheelRF.motorTorque = motorTorque;
			wheelLB.motorTorque = motorTorque;
			wheelRB.motorTorque = motorTorque;
		} else {
			wheelLF.motorTorque = 0;
			wheelRF.motorTorque = 0;
			wheelLB.motorTorque = 0;
			wheelRB.motorTorque = 0;
		}
	}

	void UpdateSounds() {
		if (startMotor.isPlaying) return;
		if (minSpeed > 0) {
			if (!idle.isPlaying) idle.Play();
			idle.pitch = 0.5f + Mathf.Clamp((body.velocity.magnitude) / 10f, 0f, 1f);
		} else {
			startMotor.Stop();
			idle.Stop();
		}
	}
}
