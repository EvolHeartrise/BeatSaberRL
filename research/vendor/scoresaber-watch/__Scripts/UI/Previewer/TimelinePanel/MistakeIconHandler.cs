using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MistakeIconHandler : MonoBehaviour
{
    private const int DeferredIconGenerationFrames = 2;

    [SerializeField] private MistakeIcon iconPrefab;
    [SerializeField] private RectTransform iconParent;

    [Space]
    [SerializeField] private Sprite badcutSprite;
    [SerializeField] private Sprite missSprite;
    [SerializeField] private Sprite bombSprite;
    [SerializeField] private Sprite wallSprite;
    [SerializeField] private Sprite pauseSprite;
    [SerializeField] private Sprite failSprite;

    [Space]
    [SerializeField] private Color badcutColor = Color.red;
    [SerializeField] private Color missColor = Color.white;
    [SerializeField] private Color bombColor = Color.yellow;
    [SerializeField] private Color wallColor = Color.magenta;
    [SerializeField] private Color pauseColor = Color.cyan;
    [SerializeField] private Color failColor = Color.grey;

    [Space]
    [SerializeField] private string badcutTooltip;
    [SerializeField] private string missTooltip;
    [SerializeField] private string bombTooltip;
    [SerializeField] private string wallTooltip;
    [SerializeField] private string failTooltip;

    private List<MistakeIcon> icons = new List<MistakeIcon>();
    private readonly List<MistakeIcon> iconPool = new List<MistakeIcon>();
    private Canvas parentCanvas;
    private bool mistakeIcons;
    private int iconGenerationRequest;
    private Coroutine iconGenerationCoroutine;


    private string GetTimeString(float time)
    {
        int totalSeconds = Mathf.FloorToInt(time);
        int seconds = totalSeconds % 60;

        string secondsString = seconds >= 10 ? $"{seconds}" : $"0{seconds}";
        return $"{totalSeconds / 60}:{secondsString}";
    }


    private void SetIconProperties(ref MistakeIcon icon, ScoringEvent scoringEvent)
    {
        string timeString = GetTimeString(scoringEvent.ObjectTime);

        icon.SetParentReferences(iconParent, parentCanvas);
        icon.SetTime(scoringEvent.ObjectTime);

        if(scoringEvent.IsWall)
        {
            icon.SetVisual(wallSprite, wallColor);
            icon.SetTooltip(wallTooltip + timeString);
            return;
        }

        switch(scoringEvent.noteEventType)
        {
            default:
            case NoteEventType.bad:
                icon.SetVisual(badcutSprite, badcutColor);
                icon.SetTooltip(badcutTooltip + timeString);
                return;
            case NoteEventType.miss:
                icon.SetVisual(missSprite, missColor);
                icon.SetTooltip(missTooltip + timeString);
                return;
            case NoteEventType.bomb:
                icon.SetVisual(bombSprite, bombColor);
                icon.SetTooltip(bombTooltip + timeString);
                return;
        }
    }


    private void SetPauseIconProperties(ref MistakeIcon icon, Pause pauseEvent)
    {
        string timeString = GetTimeString(pauseEvent.time);
        string durationString = $"{Mathf.RoundToInt(pauseEvent.duration)}s";

        icon.SetParentReferences(iconParent, parentCanvas);
        icon.SetTime(pauseEvent.time);

        icon.SetVisual(pauseSprite, pauseColor);
        icon.SetTooltip($"Pause for {durationString} at {timeString}");
    }


    private void SetFailIconProperties(ref MistakeIcon icon, float failTime)
    {
        string timeString = GetTimeString(failTime);

        icon.SetParentReferences(iconParent, parentCanvas);
        icon.SetTime(failTime);

        icon.SetVisual(failSprite, failColor);
        icon.SetTooltip(failTooltip + timeString);
    }


    private MistakeIcon CreateIcon()
    {
        MistakeIcon newIcon;
        if(iconPool.Count > 0)
        {
            int lastIconIndex = iconPool.Count - 1;
            newIcon = iconPool[lastIconIndex];
            iconPool.RemoveAt(lastIconIndex);
            newIcon.transform.SetParent(iconParent, false);
            newIcon.transform.SetAsLastSibling();
            newIcon.gameObject.SetActive(true);
        }
        else
        {
            newIcon = Instantiate(iconPrefab, iconParent, false);
        }

        icons.Add(newIcon);
        return newIcon;
    }


    private void GenerateIcons()
    {
        ClearIcons();

        if(!ReplayManager.IsReplayMode)
        {
            return;
        }

        if(!mistakeIcons)
        {
            return;
        }

        MapElementList<ScoringEvent> scoringEvents = ScoreManager.ScoringEvents;

        foreach(ScoringEvent scoringEvent in scoringEvents)
        {
            if(!scoringEvent.IsWall && !scoringEvent.IsBadHit)
            {
                continue;
            }

            MistakeIcon newIcon = CreateIcon();

            SetIconProperties(ref newIcon, scoringEvent);
        }

        foreach(Pause pauseEvent in ReplayManager.CurrentReplay.pauses)
        {
            MistakeIcon newIcon = CreateIcon();

            SetPauseIconProperties(ref newIcon, pauseEvent);
        }

        if(ReplayManager.Failed)
        {
            MistakeIcon newIcon = CreateIcon();

            SetFailIconProperties(ref newIcon, ReplayManager.FailTime);
        }
    }


    private void ScheduleIconGeneration()
    {
        iconGenerationRequest++;
        if(iconGenerationCoroutine != null)
        {
            StopCoroutine(iconGenerationCoroutine);
        }

        iconGenerationCoroutine = StartCoroutine(GenerateIconsDeferred(iconGenerationRequest));
    }


    private void CancelScheduledIconGeneration()
    {
        iconGenerationRequest++;
        if(iconGenerationCoroutine != null)
        {
            StopCoroutine(iconGenerationCoroutine);
            iconGenerationCoroutine = null;
        }
    }


    private IEnumerator GenerateIconsDeferred(int request)
    {
        for(int i = 0; i < DeferredIconGenerationFrames; i++)
        {
            yield return null;
        }

        if(request != iconGenerationRequest)
        {
            yield break;
        }

        iconGenerationCoroutine = null;
        GenerateIcons();
    }


    private void ClearIcons()
    {
        for(int i = 0; i < icons.Count; i++)
        {
            MistakeIcon icon = icons[i];
            icon.gameObject.SetActive(false);
            iconPool.Add(icon);
        }
        icons.Clear();
    }


    private void UpdateReplayMode(bool replayMode)
    {
        CancelScheduledIconGeneration();
        GenerateIcons();
    }


    private void UpdateSettings(string setting)
    {
        if(setting == "all" || setting == "mistakeicons")
        {
            mistakeIcons = SettingsManager.Loaded && SettingsManager.GetBool("mistakeicons");
            CancelScheduledIconGeneration();
            GenerateIcons();
        }
    }


    private void UpdateDifficulty(Difficulty newDifficulty) => ScheduleIconGeneration();


    private void OnEnable()
    {
        if(!parentCanvas)
        {
            parentCanvas = GetComponentInParent<Canvas>();
        }

        ReplayManager.OnReplayModeChanged += UpdateReplayMode;
        SettingsManager.OnSettingsUpdated += UpdateSettings;
        BeatmapManager.OnBeatmapDifficultyChanged += UpdateDifficulty;

        UpdateSettings("all");
    }


    private void OnDisable()
    {
        ReplayManager.OnReplayModeChanged -= UpdateReplayMode;
        SettingsManager.OnSettingsUpdated -= UpdateSettings;
        BeatmapManager.OnBeatmapDifficultyChanged -= UpdateDifficulty;

        CancelScheduledIconGeneration();
        ClearIcons();
    }
}
