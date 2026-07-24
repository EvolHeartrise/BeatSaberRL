using UnityEngine;

public class StaticLightsHint : MonoBehaviour
{
    private const string StaticLightsSetting = "staticlights";
    private const string DismissedSetting = "staticlightshintdismissed";

    [SerializeField] private RectTransform content;


    public void Dismiss()
    {
        SettingsManager.SetRule(DismissedSetting, true);
        SetVisible(false);
    }


    private void SetVisible(bool visible)
    {
        content.gameObject.SetActive(visible);
    }


    private void UpdateVisibility(string setting)
    {
        if(setting != "all" && setting != StaticLightsSetting && setting != DismissedSetting)
        {
            return;
        }

        if(!SettingsManager.Loaded)
        {
            SetVisible(false);
            return;
        }

        bool staticLights = SettingsManager.GetBool(StaticLightsSetting);
        bool dismissed = SettingsManager.GetBool(DismissedSetting);

        if(!staticLights)
        {
            if(!dismissed)
            {
                SettingsManager.SetRule(DismissedSetting, true);
            }

            SetVisible(false);
            return;
        }

        SetVisible(!dismissed);
    }


    private void OnEnable()
    {
        SettingsManager.OnSettingsUpdated += UpdateVisibility;
        UpdateVisibility("all");
    }


    private void OnDisable()
    {
        SettingsManager.OnSettingsUpdated -= UpdateVisibility;
    }
}
