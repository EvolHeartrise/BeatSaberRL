using System.Web;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class SharePanel : MonoBehaviour
{
    public bool UseTimestamp = false;

    [SerializeField] private TMP_InputField urlOutput;
    [SerializeField] private Toggle timeStampToggle;
    [SerializeField] private TextMeshProUGUI timeStampToggleLabel;

    private bool hasCachedUrlState;
    private ulong cachedUrlTime = ulong.MaxValue;
    private bool cachedUrlUseTimestamp;
    private bool cachedUrlReplayMode;
    private bool cachedUrlIgnoreMapForSharing;
    private string cachedUrlSSScoreId;
    private string cachedUrlBLReplayID;
    private string cachedUrlReplayURL;
    private string cachedUrlMapID;
    private string cachedUrlMapURL;
    private DifficultyCharacteristic? cachedUrlCharacteristic;
    private DifficultyRank? cachedUrlDiffRank;
    private ulong cachedToggleLabelTime = ulong.MaxValue;


    public void SetEnableTimestamp(bool timestamp)
    {
        UseTimestamp = timestamp;
    }


    private void UpdateText(ulong currentTime)
    {
        ulong urlTime = UseTimestamp ? currentTime : ulong.MaxValue;
        bool replayMode = ReplayManager.IsReplayMode;
        bool ignoreMapForSharing = UrlArgHandler.ignoreMapForSharing;
        string loadedSSScoreId = UrlArgHandler.LoadedSSScoreId;
        string loadedBLReplayID = UrlArgHandler.LoadedBLReplayID;
        string loadedReplayURL = UrlArgHandler.LoadedReplayURL;
        string loadedMapID = UrlArgHandler.LoadedMapID;
        string loadedMapURL = UrlArgHandler.LoadedMapURL;
        DifficultyCharacteristic? loadedCharacteristic = UrlArgHandler.LoadedCharacteristic;
        DifficultyRank? loadedDiffRank = UrlArgHandler.LoadedDiffRank;

        if(hasCachedUrlState
            && cachedUrlTime == urlTime
            && cachedUrlUseTimestamp == UseTimestamp
            && cachedUrlReplayMode == replayMode
            && cachedUrlIgnoreMapForSharing == ignoreMapForSharing
            && cachedUrlSSScoreId == loadedSSScoreId
            && cachedUrlBLReplayID == loadedBLReplayID
            && cachedUrlReplayURL == loadedReplayURL
            && cachedUrlMapID == loadedMapID
            && cachedUrlMapURL == loadedMapURL
            && cachedUrlCharacteristic == loadedCharacteristic
            && cachedUrlDiffRank == loadedDiffRank)
        {
            return;
        }

        hasCachedUrlState = true;
        cachedUrlTime = urlTime;
        cachedUrlUseTimestamp = UseTimestamp;
        cachedUrlReplayMode = replayMode;
        cachedUrlIgnoreMapForSharing = ignoreMapForSharing;
        cachedUrlSSScoreId = loadedSSScoreId;
        cachedUrlBLReplayID = loadedBLReplayID;
        cachedUrlReplayURL = loadedReplayURL;
        cachedUrlMapID = loadedMapID;
        cachedUrlMapURL = loadedMapURL;
        cachedUrlCharacteristic = loadedCharacteristic;
        cachedUrlDiffRank = loadedDiffRank;

        string newText = UrlArgHandler.ArcViewerURL;

        if(replayMode)
        {
            if(!string.IsNullOrEmpty(loadedSSScoreId))
            {
                newText += $"?ssScoreId={loadedSSScoreId}";
            }
            else if(!string.IsNullOrEmpty(loadedBLReplayID))
            {
                newText += $"?scoreID={loadedBLReplayID}";
            }
            else if(!string.IsNullOrEmpty(loadedReplayURL))
            {
                newText += $"?replayURL={HttpUtility.UrlEncode(loadedReplayURL)}";
            }
            else
            {
                //The replay was loaded locally, this menu shouldn't be open
                gameObject.SetActive(false);
                return;
            }

            if(!ignoreMapForSharing && !string.IsNullOrEmpty(loadedMapURL))
            {
                //Include custom set map for replays
                newText += $"&url={HttpUtility.UrlEncode(loadedMapURL)}";
            }
        }
        else
        {
            if(!string.IsNullOrEmpty(loadedMapID))
            {
                newText += $"?id={loadedMapID}";
            }
            else if(!string.IsNullOrEmpty(loadedMapURL))
            {
                newText += $"?url={HttpUtility.UrlEncode(loadedMapURL)}";
            }
            else
            {
                //The map was loaded locally, this menu shouldn't be open
                gameObject.SetActive(false);
                return;
            }

            //Only include difficulty arguments outside of replays
            string mode = loadedCharacteristic?.ToString() ?? "";
            string difficulty = loadedDiffRank?.ToString() ?? "";

            if(!string.IsNullOrEmpty(mode))
            {
                newText += $"&mode={mode}";
            }
            if(!string.IsNullOrEmpty(difficulty))
            {
                newText += $"&difficulty={difficulty}";
            }
        }

        if(UseTimestamp)
        {
            float time = currentTime;
            newText += $"&t={time}";
        }

        if(newText != urlOutput.text)
        {
            urlOutput.text = newText;
        }
    }


    private void UpdateToggleLabel(ulong currentTime)
    {
        if(cachedToggleLabelTime == currentTime)
        {
            return;
        }

        cachedToggleLabelTime = currentTime;
        ulong currentSeconds = currentTime % 60;
        ulong currentMinutes = currentTime / 60;

        string secondsString = currentSeconds >= 10 ? $"{currentSeconds}" : $"0{currentSeconds}";

        string time = $"{currentMinutes}:{secondsString}";

        timeStampToggleLabel.text = $"Start at {time}";
    }


    private void Update()
    {
        ulong currentTime = (ulong)TimeManager.CurrentTime;

        UpdateText(currentTime);
        UpdateToggleLabel(currentTime);
    }


    private void OnEnable()
    {
        timeStampToggle.SetIsOnWithoutNotify(UseTimestamp);
    }
}
