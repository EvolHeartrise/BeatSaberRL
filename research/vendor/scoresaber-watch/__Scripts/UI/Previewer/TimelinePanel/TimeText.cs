using UnityEngine;
using TMPro;

public class TimeText : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI timeText;
    [SerializeField] private TextMeshProUGUI beatText;

    private int cachedTotalSeconds = int.MinValue;
    private int cachedFlooredBeat = int.MinValue;


    public void UpdateText(float beat)
    {
        int totalSeconds = Mathf.FloorToInt(TimeManager.CurrentTime);
        if(cachedTotalSeconds != totalSeconds)
        {
            cachedTotalSeconds = totalSeconds;

            int currentSeconds = totalSeconds % 60;
            int currentMinutes = totalSeconds / 60;
            string secondsString = currentSeconds >= 10 ? $"{currentSeconds}" : $"0{currentSeconds}";

            timeText.text = $"{currentMinutes}:{secondsString}";
        }

        int flooredBeat = Mathf.FloorToInt(beat);
        if(cachedFlooredBeat != flooredBeat)
        {
            cachedFlooredBeat = flooredBeat;

            beatText.text = flooredBeat.ToString();
        }
    }


    private void OnEnable()
    {
        cachedTotalSeconds = int.MinValue;
        cachedFlooredBeat = int.MinValue;

        TimeManager.OnBeatChanged += UpdateText;
    }


    private void OnDisable()
    {
        TimeManager.OnBeatChanged -= UpdateText;
    }
}