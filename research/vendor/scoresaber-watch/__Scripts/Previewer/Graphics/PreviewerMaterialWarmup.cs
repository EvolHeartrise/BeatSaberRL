using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PreviewerMaterialWarmup : MonoBehaviour
{
    private static bool started;
    private static Mesh warmupMesh;

    private ObjectManager objectManager;


    public static void RunOnce(ObjectManager manager)
    {
        if(started || manager == null)
        {
            return;
        }

        started = true;
        PreviewerMaterialWarmup warmup = manager.GetComponent<PreviewerMaterialWarmup>();
        if(warmup == null)
        {
            warmup = manager.gameObject.AddComponent<PreviewerMaterialWarmup>();
        }

        warmup.objectManager = manager;
        warmup.StartCoroutine(warmup.WarmupNextFrame());
    }


    private IEnumerator WarmupNextFrame()
    {
        yield return null;

        if(objectManager == null)
        {
            Destroy(this);
            yield break;
        }

        GameObject warmupRoot = new GameObject("Previewer Material Warmup");
        warmupRoot.hideFlags = HideFlags.HideAndDontSave;
        warmupRoot.transform.position = new Vector3(5000f, 5000f, 5000f);

        List<Action> cleanups = new List<Action>();
        try
        {
            AddMapObjectWarmups(warmupRoot.transform, cleanups);
            AddScoreIndicatorWarmup(warmupRoot.transform, cleanups);
            AddTrailMaterialWarmups(warmupRoot.transform, cleanups);

            RenderWarmup(warmupRoot.transform.position);
        }
        catch(Exception err)
        {
            Debug.LogWarning($"Previewer material warmup skipped with error: {err.Message}");
        }
        finally
        {
            for(int i = cleanups.Count - 1; i >= 0; i--)
            {
                cleanups[i]();
            }

            Destroy(warmupRoot);
            Destroy(this);
        }
    }


    private void AddMapObjectWarmups(Transform parent, List<Action> cleanups)
    {
        if(objectManager.noteManager != null)
        {
            NoteHandler noteHandler = objectManager.noteManager.CreateWarmupVisual(parent, new Vector3(-1.2f, 0.45f, 0f), true);
            if(noteHandler != null)
            {
                cleanups.Add(() => objectManager.noteManager.ReleaseWarmupVisual(noteHandler));
            }
        }

        if(objectManager.chainManager != null)
        {
            ChainLinkHandler chainLinkHandler = objectManager.chainManager.CreateWarmupVisual(parent, new Vector3(-0.4f, 0.45f, 0f));
            if(chainLinkHandler != null)
            {
                cleanups.Add(() => objectManager.chainManager.ReleaseWarmupVisual(chainLinkHandler));
            }
        }

        if(objectManager.arcManager != null)
        {
            ArcHandler arcHandler = objectManager.arcManager.CreateWarmupVisual(parent, new Vector3(0.4f, 0.45f, 0f));
            if(arcHandler != null)
            {
                cleanups.Add(() => objectManager.arcManager.ReleaseWarmupVisual(arcHandler));
            }
        }

        if(objectManager.wallManager != null)
        {
            WallHandler wallHandler = objectManager.wallManager.CreateWarmupVisual(parent, new Vector3(1.2f, 0.45f, 0f));
            if(wallHandler != null)
            {
                cleanups.Add(() => objectManager.wallManager.ReleaseWarmupVisual(wallHandler));
            }
        }
    }


    private void AddScoreIndicatorWarmup(Transform parent, List<Action> cleanups)
    {
        TMProPool pool = FindSceneObject<TMProPool>();
        if(pool == null)
        {
            return;
        }

        ScoreIndicatorHandler scoreIndicator = pool.GetObject();
        scoreIndicator.transform.SetParent(parent);
        scoreIndicator.transform.localPosition = new Vector3(-0.6f, -0.55f, 0f);
        scoreIndicator.transform.localRotation = Quaternion.identity;
        scoreIndicator.transform.localScale = Vector3.one;
        scoreIndicator.gameObject.SetActive(true);
        scoreIndicator.SetIconActive(false);
        scoreIndicator.SetText("100");
        scoreIndicator.SetColor(Color.white);

        cleanups.Add(() => pool.ReleaseObject(scoreIndicator));
    }


    private void AddTrailMaterialWarmups(Transform parent, List<Action> cleanups)
    {
        SaberHandler[] sabers = Resources.FindObjectsOfTypeAll<SaberHandler>();
        HashSet<Material> warmedMaterials = new HashSet<Material>();
        int materialCount = 0;
        for(int i = 0; i < sabers.Length; i++)
        {
            SaberHandler saber = sabers[i];
            if(!IsSceneObject(saber))
            {
                continue;
            }

            MeshRenderer[] renderers = saber.GetComponentsInChildren<MeshRenderer>(true);
            for(int j = 0; j < renderers.Length; j++)
            {
                MeshRenderer sourceRenderer = renderers[j];
                Material material = sourceRenderer.sharedMaterial;
                if(!IsTrailMaterial(sourceRenderer, material) || warmedMaterials.Contains(material))
                {
                    continue;
                }

                warmedMaterials.Add(material);
                GameObject materialObject = CreateMaterialWarmupObject(parent, material, materialCount);
                cleanups.Add(() => Destroy(materialObject));
                materialCount++;
            }
        }
    }


    private GameObject CreateMaterialWarmupObject(Transform parent, Material material, int index)
    {
        GameObject materialObject = new GameObject("Previewer Trail Material Warmup");
        materialObject.name = "Previewer Trail Material Warmup";
        materialObject.hideFlags = HideFlags.HideAndDontSave;
        materialObject.transform.SetParent(parent);
        materialObject.transform.localPosition = new Vector3(0.2f + (index * 0.45f), -0.55f, 0f);
        materialObject.transform.localRotation = Quaternion.identity;
        materialObject.transform.localScale = Vector3.one * 0.25f;

        MeshFilter filter = materialObject.AddComponent<MeshFilter>();
        filter.sharedMesh = GetWarmupMesh();

        MeshRenderer renderer = materialObject.AddComponent<MeshRenderer>();
        renderer.sharedMaterial = material;

        MaterialPropertyBlock properties = new MaterialPropertyBlock();
        properties.SetColor("_BaseColor", NoteManager.LeftSaberColor);
        if(material.HasProperty("_Brightness"))
        {
            properties.SetFloat("_Brightness", Mathf.Clamp(SettingsManager.GetFloat("sabertrailbrightness"), 0f, 2f));
        }
        if(material.HasProperty("_TrailTexture"))
        {
            properties.SetTexture("_TrailTexture", Texture2D.whiteTexture);
        }
        renderer.SetPropertyBlock(properties);

        return materialObject;
    }


    private static Mesh GetWarmupMesh()
    {
        if(warmupMesh != null)
        {
            return warmupMesh;
        }

        warmupMesh = new Mesh
        {
            name = "Previewer Material Warmup Mesh",
            hideFlags = HideFlags.HideAndDontSave,
            vertices = new[]
            {
                new Vector3(-0.5f, -0.5f, 0f),
                new Vector3(-0.5f, 0.5f, 0f),
                new Vector3(0.5f, 0.5f, 0f),
                new Vector3(0.5f, -0.5f, 0f)
            },
            uv = new[]
            {
                new Vector2(0f, 0f),
                new Vector2(0f, 1f),
                new Vector2(1f, 1f),
                new Vector2(1f, 0f)
            },
            triangles = new[] { 0, 1, 2, 0, 2, 3 }
        };
        warmupMesh.RecalculateBounds();
        return warmupMesh;
    }


    private void RenderWarmup(Vector3 targetPosition)
    {
        RenderTexture targetTexture = RenderTexture.GetTemporary(32, 32, 16);
        GameObject cameraObject = new GameObject("Previewer Material Warmup Camera");
        cameraObject.hideFlags = HideFlags.HideAndDontSave;

        try
        {
            Camera camera = cameraObject.AddComponent<Camera>();
            camera.enabled = false;
            camera.orthographic = true;
            camera.orthographicSize = 2f;
            camera.nearClipPlane = 0.01f;
            camera.farClipPlane = 16f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = Color.clear;
            camera.targetTexture = targetTexture;
            camera.transform.position = targetPosition + new Vector3(0f, 0f, -6f);
            camera.transform.rotation = Quaternion.identity;
            camera.Render();
            camera.targetTexture = null;
        }
        finally
        {
            RenderTexture.ReleaseTemporary(targetTexture);
            Destroy(cameraObject);
        }
    }


    private static T FindSceneObject<T>() where T : Component
    {
        T[] objects = Resources.FindObjectsOfTypeAll<T>();
        for(int i = 0; i < objects.Length; i++)
        {
            T target = objects[i];
            if(IsSceneObject(target))
            {
                return target;
            }
        }

        return null;
    }


    private static bool IsSceneObject(Component component)
    {
        return component != null && component.gameObject.scene.IsValid();
    }


    private static bool IsTrailMaterial(Renderer renderer, Material material)
    {
        if(material == null)
        {
            return false;
        }

        if(material.HasProperty("_TrailTexture"))
        {
            return true;
        }

        return renderer.gameObject.name.ToLowerInvariant().Contains("trail")
            || material.name.ToLowerInvariant().Contains("trail");
    }
}
