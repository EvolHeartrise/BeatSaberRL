using UnityEngine;

public class LinkButton : MonoBehaviour
{
    [SerializeField] private string url;


    public void OpenLink()
    {
        ExternalLinkOpener.Open(url);
    }
}