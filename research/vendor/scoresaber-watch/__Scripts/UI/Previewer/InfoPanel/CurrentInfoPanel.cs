using TMPro;
using UnityEngine;

public class CurrentInfoPanel : MonoBehaviour
{
    public const float FramerateSampleTime = 0.5f;

    [SerializeField] private TextMeshProUGUI bpmText;
    [SerializeField] private TextMeshProUGUI njsText;
    [SerializeField] private TextMeshProUGUI fpsCounter;
    [SerializeField] private TextMeshProUGUI replayFpsCounter;

    [Space]
    [SerializeField] private float previewModeY;
    [SerializeField] private float replayModeY;

    private bool showCurrentMapStats;
    private bool showFPS;
    private bool showReplayFPS => showFPS && ReplayManager.IsReplayMode && UIStateManager.CurrentState == UIState.Previewer;

    private int checkedFrameCount;
    private float timeSinceFramerateUpdate;
    private int cachedReplayFPS = int.MinValue;


    private void UpdateCurrentBPM()
    {
        string bpm = TimeManager.CurrentBPM.Round(3).ToString();
        bpmText.text = $"BPM: {bpm}";
    }


    private void UpdateCurrentNJS()
    {
        string njs = ObjectManager.Instance.jumpManager.NJS.Round(3).ToString();
        njsText.text = $"NJS: {njs}";
    }


    private void SetCountersActive()
    {
        fpsCounter.gameObject.SetActive(showFPS);
        replayFpsCounter.gameObject.SetActive(showReplayFPS);

        checkedFrameCount = 0;
        timeSinceFramerateUpdate = 0f;
        cachedReplayFPS = int.MinValue;
    }


    private void UpdateReplayMode(bool replayMode) => SetCountersActive();


    private void UpdateUIState(UIState newState) => SetCountersActive();


    private void UpdateSettings(string setting)
    {
        if(setting == "all" || setting == "showcurrentstats")
        {
            showCurrentMapStats = SettingsManager.GetBool("showcurrentstats");
            bpmText.gameObject.SetActive(showCurrentMapStats);
            njsText.gameObject.SetActive(showCurrentMapStats);
        }

        if(setting == "all" || setting == "fpscounter")
        {
            showFPS = SettingsManager.GetBool("fpscounter");
            SetCountersActive();
        }
    }


    private void UpdateBeat(float beat)
    {
        if(showReplayFPS)
        {
            int averageFPS = PlayerPositionManager.AverageFPS;
            if(cachedReplayFPS != averageFPS)
            {
                cachedReplayFPS = averageFPS;
                replayFpsCounter.text = "Replay: " + averageFPS;
            }
        }
    }


    private void OnEnable()
    {
        SettingsManager.OnSettingsUpdated += UpdateSettings;
        ReplayManager.OnReplayModeChanged += UpdateReplayMode;
        UIStateManager.OnUIStateChanged += UpdateUIState;
        TimeManager.OnBeatChanged += UpdateBeat;

        if(SettingsManager.Loaded)
        {
            UpdateSettings("all");
        }

        // adjust y position to account for the player info panel being enabled/disabled
        RectTransform rectTransform = (RectTransform)transform;
        float yPos = ReplayManager.IsReplayMode ? replayModeY : previewModeY;
        rectTransform.anchoredPosition = new Vector2(rectTransform.anchoredPosition.x, yPos);

        if(showCurrentMapStats)
        {
            UpdateCurrentBPM();
            UpdateCurrentNJS();
        }
    }


    private void OnDisable()
    {
        SettingsManager.OnSettingsUpdated -= UpdateSettings;
        ReplayManager.OnReplayModeChanged -= UpdateReplayMode;
        UIStateManager.OnUIStateChanged -= UpdateUIState;
        TimeManager.OnBeatChanged -= UpdateBeat;
    }


    private void Update()
    {
        if(showFPS)
        {
            timeSinceFramerateUpdate += Time.deltaTime;
            checkedFrameCount++;

            if(timeSinceFramerateUpdate >= FramerateSampleTime)
            {
                float averageFramerate = 1f / (timeSinceFramerateUpdate / checkedFrameCount);
                int fps = Mathf.RoundToInt(averageFramerate);
                fpsCounter.text = "FPS: " + fps;

                checkedFrameCount = 0;
                timeSinceFramerateUpdate = 0f;
            }
        }
    }


    private void LateUpdate()
    {
        if(showCurrentMapStats)
        {
            UpdateCurrentBPM();
            UpdateCurrentNJS();
        }
    }
}
