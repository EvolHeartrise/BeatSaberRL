using UnityEngine;

public class BombManager : MapElementManager<Bomb>
{
    [SerializeField] private ObjectPool<BombHandler> bombPool;

    [Space]
    [SerializeField] private Material complexMaterial;
    [SerializeField] private Material simpleMaterial;

    private Material complexRuntimeMaterial;
    private Material simpleRuntimeMaterial;

    private static bool playBadCutSound => TimeManager.Playing && SettingsManager.GetBool("usebadhitsound") && HitSoundManager.HitSoundsEnabled;


    public void ReloadBombs()
    {
        CustomRTObjects.Clear();
        CustomRTObjects.GetTime = GetSpawnTime;
        for(int i = Objects.Count - 1; i >= 0; i--)
        {
            Bomb b = Objects[i];
            if(b.CustomRT != null)
            {
                Objects.Remove(b);
                CustomRTObjects.Add(b);
            }
        }
        CustomRTObjects.SortElementsByBeat();

        Objects.ResetStartIndex();
        CustomRTObjects.ResetStartIndex();

        ClearRenderedVisuals();
        ClearSharedMaterialCache();
        UpdateVisuals();
    }


    public override void UpdateVisual(Bomb b)
    {
        float reactionTime = b.CustomRT ?? jumpManager.ReactionTime;
        float njs = b.CustomNJS != null
            ? jumpManager.GetAdjustedNJS((float)b.CustomNJS, reactionTime)
            : jumpManager.EffectiveNJS;
        float halfJumpDistance = jumpManager.WorldSpaceFromTime(reactionTime, njs);

        float worldDist = jumpManager.GetZPosition(b.Time, njs, reactionTime, halfJumpDistance);
        Vector3 worldPos = new Vector3(b.Position.x, b.Position.y, worldDist);

        worldPos.y += objectManager.playerHeightOffset;

        if(objectManager.doMovementAnimation)
        {
            worldPos.y = jumpManager.GetObjectY(b.StartY, worldPos.y, worldDist, halfJumpDistance, b.Time, reactionTime);
        }

        Transform visualTransform;
        if(b.Visual == null)
        {
            b.BombHandler = bombPool.GetObject();
            b.Visual = b.BombHandler.gameObject;
            b.source = b.BombHandler.audioSource;

            visualTransform = b.Visual.transform;
            visualTransform.SetParent(transform);
            b.BombHandler.EnableVisual();

            b.BombHandler.SetMaterial(GetSharedMaterial());

            if(SettingsManager.GetBool("chromaobjectcolors") && b.CustomColor != null)
            {
                //This bomb has a unique chroma color
                b.BombHandler.SetProperties(b.CustomMaterialProperties);
            }

            if(b.WasHit && playBadCutSound)
            {
                //This bomb should play
                HitSoundManager.ScheduleHitsound(b);
            }

            if(ReplayManager.IsReplayMode && b.WasBadCut && SettingsManager.GetBool("highlighterrors"))
            {
                b.BombHandler.SetOutline(true, SettingsManager.GetColor("badcutoutlinecolor"));
            }
            else b.BombHandler.SetOutline(false);

            b.Visual.SetActive(true);
            RenderedObjects.Add(b);
        }
        else
        {
            visualTransform = b.Visual.transform;
        }
        visualTransform.localPosition = worldPos;
    }


    public override float GetSpawnTime(Bomb b)
    {
        return b.Time - (float)b.CustomRT - objectManager.moveTime;
    }


    public override bool VisualInSpawnRange(Bomb b)
    {
        return jumpManager.CheckInSpawnRange(b.Time, b.CustomRT ?? jumpManager.ReactionTime, true, true, b.HitOffset);
    }


    public override void ReleaseVisual(Bomb b)
    {
        b.source.Stop();
        bombPool.ReleaseObject(b.BombHandler);

        b.Visual = null;
        b.source = null;
        b.BombHandler = null;
    }


    public override void ClearOutsideVisuals()
    {
        for(int i = RenderedObjects.Count - 1; i >= 0; i--)
        {
            Bomb b = RenderedObjects[i];
            if(!jumpManager.CheckInSpawnRange(b.Time, b.CustomRT ?? jumpManager.ReactionTime, !b.WasHit, true, b.HitOffset))
            {
                if(b.source.isPlaying || (ReplayManager.IsReplayMode && b.Time > TimeManager.CurrentTime && b.Time < TimeManager.CurrentTime + 0.5f))
                {
                    //Only clear the visual elements if the hitsound is still playing
                    b.BombHandler.DisableVisual();
                }
                else
                {
                    ReleaseVisual(b);
                    RenderedObjects.RemoveAt(i);
                }
            }
            else if(b.ShouldShowVisual) b.BombHandler.EnableVisual();
            else b.BombHandler.DisableVisual();
        }
    }


    public override void UpdateObjects(MapElementList<Bomb> objects)
    {
        if(objects.Count == 0)
        {
            return;
        }

        int startIndex = GetStartIndex(TimeManager.CurrentTime, objects);
        if(startIndex < 0)
        {
            return;
        }

        for(int i = startIndex; i < objects.Count; i++)
        {
            Bomb b = objects[i];
            if(jumpManager.CheckInSpawnRange(b.Time, b.CustomRT ?? jumpManager.ReactionTime, !b.WasHit, true, b.HitOffset))
            {
                UpdateVisual(b);
                if(!b.ShouldShowVisual)
                {
                    b.BombHandler.DisableVisual();
                }
            }
            else if(!VisualInSpawnRange(b))
            {
                break;
            }
        }
    }


    public void RescheduleHitsounds()
    {
        foreach(Bomb b in RenderedObjects)
        {
#if !UNITY_WEBGL || UNITY_EDITOR
            if(b.WasHit && playBadCutSound && b.source != null)
#else
            if(b.WasHit && playBadCutSound)
#endif
            {
                HitSoundManager.ScheduleHitsound(b);
            }
        }
    }


    private Material GetSharedMaterial()
    {
        if(objectManager.useSimpleBombMaterial)
        {
            if(simpleRuntimeMaterial == null && simpleMaterial != null)
            {
                simpleRuntimeMaterial = new Material(simpleMaterial);
            }

            return simpleRuntimeMaterial;
        }

        if(complexRuntimeMaterial == null && complexMaterial != null)
        {
            complexRuntimeMaterial = new Material(complexMaterial);
        }

        return complexRuntimeMaterial;
    }


    private void ClearSharedMaterialCache()
    {
        if(complexRuntimeMaterial != null) Destroy(complexRuntimeMaterial);
        if(simpleRuntimeMaterial != null) Destroy(simpleRuntimeMaterial);

        complexRuntimeMaterial = null;
        simpleRuntimeMaterial = null;
    }


    private void OnDestroy()
    {
        ClearSharedMaterialCache();
    }
}


public class Bomb : HitSoundEmitter
{
    public float StartY;

    public int ScoreEventID;

    public BombHandler BombHandler;
    public MaterialPropertyBlock CustomMaterialProperties;

    public Bomb(BeatmapBombNote b)
    {
        Vector2 position = ObjectManager.CalculateObjectPosition(b.x, b.y, b.customData?.coordinates);

        Beat = b.b;
        Position = position;

        WasHit = false;
        WasBadCut = false;
        HitOffset = 0f;

        if(b.customData != null)
        {
            if(b.customData.color != null)
            {
                CustomColor = ColorManager.ColorFromCustomDataColor(b.customData.color);

                CustomMaterialProperties = new MaterialPropertyBlock();
                CustomMaterialProperties.SetColor("_BaseColor", (Color)CustomColor);
            }

            CustomNJS = b.customData.noteJumpMovementSpeed;
            if(b.customData.noteJumpStartBeatOffset != null)
            {
                CustomRT = BeatmapManager.GetCustomRT((float)b.customData.noteJumpStartBeatOffset);
            }
        }
    }
}
