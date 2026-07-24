using System;
using System.Collections;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;

public class ExplorerPathLoader : MonoBehaviour
{
    public static string PreviousExplorerPath { get; private set; }

#if !UNITY_WEBGL || UNITY_EDITOR
    private const string explorerPathFileName = "ExplorerPath.txt";
    private static string explorerPathFile => Path.Combine(Application.persistentDataPath, explorerPathFileName);

    private Coroutine saveCoroutine;
    private bool saving = false;


    private async Task SavePreviousPath()
    {
        try
        {
            await File.WriteAllTextAsync(explorerPathFile, PreviousExplorerPath);
        }
        catch(Exception err)
        {
            Debug.LogWarning($"Failed to write previous path file with error: {err.Message}, {err.StackTrace}");
        }
    }


    private IEnumerator SavePreviousPathCoroutine()
    {
        saving = true;

        using Task saveTask = SavePreviousPath();
        yield return new WaitUntil(() => saveTask.IsCompleted);

        saving = false;
    }


    public void SetPreviousPath(string previousPath)
    {
        if(!Directory.Exists(previousPath))
        {
            Debug.LogWarning($"Path {previousPath} is not a valid directory!");
            PreviousExplorerPath = "";
        }
        else PreviousExplorerPath = previousPath;

        if(saving)
        {
            StopCoroutine(saveCoroutine);
        }

        saveCoroutine = StartCoroutine(SavePreviousPathCoroutine());
    }


    private async Task LoadExplorerPath()
    {
        try
        {
            string filePath = explorerPathFile;
            if(!File.Exists(filePath))
            {
                PreviousExplorerPath = "";
                return;
            }

            PreviousExplorerPath = await File.ReadAllTextAsync(filePath);
            if(PreviousExplorerPath != "" && !Directory.Exists(PreviousExplorerPath))
            {
                Debug.LogWarning($"Previous path {PreviousExplorerPath} is not a valid directory!");
                PreviousExplorerPath = "";
            }
        }
        catch(Exception err)
        {
            Debug.LogWarning($"Failed to load previous explorer path with error: {err.Message}, {err.StackTrace}");
            PreviousExplorerPath = "";
        }
    }


    private async void Start()
    {
        await LoadExplorerPath();
    }
#endif
}