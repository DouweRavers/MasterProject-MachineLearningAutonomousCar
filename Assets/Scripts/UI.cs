using UnityEngine;
using UnityEngine.SceneManagement;

public class UI : MonoBehaviour {
	public static int player = 1;
	public TMPro.TMP_Dropdown dropdown;

	void Start() {
		if (dropdown != null) dropdown.value = player;
	}

	public void PlayCourse(int level) {
		SceneManager.LoadScene(level);
	}

	public void Menu() {
		SceneManager.LoadScene(0);
	}

	public void Quit() {
		Application.Quit();
	}

	public void SetPlayer(int value) {
		player = value;
	}

}
