using UnityEngine;
using UnityEngine.SceneManagement;

public class UI : MonoBehaviour {
	public void PlayCourse(int level) {
		SceneManager.LoadScene(level);
	}

	public void Menu() {
		SceneManager.LoadScene(0);
	}

	public void Quit() {
		Application.Quit();
	}

}
