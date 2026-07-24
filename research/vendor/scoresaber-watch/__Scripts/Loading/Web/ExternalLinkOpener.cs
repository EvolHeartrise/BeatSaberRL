using System.Runtime.InteropServices;
using UnityEngine;

public static class ExternalLinkOpener
{
#if UNITY_WEBGL && !UNITY_EDITOR
    [DllImport("__Internal")]
    private static extern void OpenExternalURL(string url);
#endif

    public static void Open(string url)
    {
        if(string.IsNullOrEmpty(url)) return;

#if UNITY_WEBGL && !UNITY_EDITOR
        OpenExternalURL(url);
#else
        Application.OpenURL(url);
#endif
    }
}
