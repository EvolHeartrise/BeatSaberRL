using UnityEngine;

public class StaticLightsWarning : MonoBehaviour
{
    public void CheckShowWarning()
    {
        gameObject.SetActive(false);
    }


    public void ShowPanel()
    {
        gameObject.SetActive(false);
    }


    public void ClosePanel()
    {
        gameObject.SetActive(false);
    }


    private void Start()
    {
        CheckShowWarning();
    }
}