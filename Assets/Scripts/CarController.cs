using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class CarController : MonoBehaviour
{
	public float brakeTorque = 1000000, motorTorque = 5000, steerAngle = 15;


	public WheelCollider wheelLF, wheelRF, wheelLB, wheelRB;
	public Transform wheelVisualLF, wheelVisualRF, wheelVisualLB, wheelVisualRB;
	Rigidbody body;
	public float input;
	void Start()
	{
		// GetComponent<AudioSource>().pitch = 0.2f;
		body = GetComponent<Rigidbody>();
	}

	void Update()
	{
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

		// GetComponent<AudioSource>().pitch = 0.2f + Mathf.Clamp(GetComponent<Rigidbody>().velocity.magnitude / 80f, 0f, 0.6f);


	}

	void FixedUpdate()
	{
		wheelLF.brakeTorque = 0;
		wheelRF.brakeTorque = 0;
		wheelLB.brakeTorque = 0;
		wheelRB.brakeTorque = 0;
		if (Input.GetKey("space") || body.velocity.magnitude > 12.5f)
		{
			wheelLF.brakeTorque = brakeTorque;
			wheelRF.brakeTorque = brakeTorque;
			wheelLB.brakeTorque = brakeTorque;
			wheelRB.brakeTorque = brakeTorque;
		}

		if (Input.GetAxis("Vertical") > 0 || body.velocity.magnitude < 10)
		{
			wheelLF.motorTorque = motorTorque;
			wheelRF.motorTorque = motorTorque;
			wheelLB.motorTorque = motorTorque;
			wheelRB.motorTorque = motorTorque;
		}
		else if (Input.GetAxis("Vertical") < 0)
		{
			wheelLF.motorTorque = -1 * motorTorque;
			wheelRF.motorTorque = -1 * motorTorque;
			wheelLB.motorTorque = -1 * motorTorque;
			wheelRB.motorTorque = -1 * motorTorque;
		}
		else
		{
			wheelLF.motorTorque = 0;
			wheelRF.motorTorque = 0;
			wheelLB.motorTorque = 0;
			wheelRB.motorTorque = 0;
		}

		if (Input.GetAxis("Horizontal") > 0)
		{
			wheelLF.steerAngle = steerAngle;
			wheelRF.steerAngle = steerAngle;
			input = 1f;
		}
		else if (Input.GetAxis("Horizontal") < 0)
		{
			wheelLF.steerAngle = -1 * steerAngle;
			wheelRF.steerAngle = -1 * steerAngle;
			input = -1f;
		}
		else
		{
			wheelLF.steerAngle = 0;
			wheelRF.steerAngle = 0;
			input = 0f;
		}
	}
}
