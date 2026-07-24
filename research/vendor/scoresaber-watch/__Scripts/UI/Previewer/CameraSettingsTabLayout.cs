using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class CameraSettingsTabLayout : MonoBehaviour
{
    [SerializeField] private RectTransform contentTransform;
    [SerializeField] private Button nonActiveCameraToggle;
    [SerializeField] private RectTransform nonActiveCameraChevron;
    [SerializeField] private List<CameraSettingsSection> sections;

    private bool showNonActiveCameras;


    public void ToggleNonActiveCameras()
    {
        showNonActiveCameras = !showNonActiveCameras;
        UpdateLayout();
    }


    private CameraSettingsSectionType ActiveSectionType()
    {
        if(CameraUpdater.Freecam)
        {
            return CameraSettingsSectionType.Free;
        }

        if(ReplayManager.IsReplayMode)
        {
            if(SettingsManager.Loaded && SettingsManager.GetBool("firstpersonreplay"))
            {
                return CameraSettingsSectionType.FirstPerson;
            }

            return CameraSettingsSectionType.Replay;
        }

        return CameraSettingsSectionType.Preview;
    }


    private void SetSectionActive(CameraSettingsSection section, bool active)
    {
        section.Header.gameObject.SetActive(active);

        foreach(RectTransform item in section.Items)
        {
            item.gameObject.SetActive(active);
        }
    }


    private int SetSectionSiblingIndex(CameraSettingsSection section, int siblingIndex)
    {
        section.Header.SetSiblingIndex(siblingIndex++);

        foreach(RectTransform item in section.Items)
        {
            item.SetSiblingIndex(siblingIndex++);
        }

        return siblingIndex;
    }


    private void UpdateLayout()
    {
        CameraSettingsSectionType activeSectionType = ActiveSectionType();
        int activeSectionIndex = sections.FindIndex(section => section.Type == activeSectionType);

        if(activeSectionIndex < 0)
        {
            return;
        }

        int siblingIndex = 0;
        CameraSettingsSection activeSection = sections[activeSectionIndex];
        SetSectionActive(activeSection, true);
        siblingIndex = SetSectionSiblingIndex(activeSection, siblingIndex);

        nonActiveCameraToggle.gameObject.SetActive(true);
        nonActiveCameraToggle.transform.SetSiblingIndex(siblingIndex++);

        for(int i = 0; i < sections.Count; i++)
        {
            if(i == activeSectionIndex)
            {
                continue;
            }

            CameraSettingsSection section = sections[i];
            SetSectionActive(section, showNonActiveCameras);
            if(showNonActiveCameras)
            {
                siblingIndex = SetSectionSiblingIndex(section, siblingIndex);
            }
        }

        LayoutRebuilder.ForceRebuildLayoutImmediate(contentTransform);
        nonActiveCameraChevron.localEulerAngles = new Vector3(0f, 0f, showNonActiveCameras ? 90f : -90f);
    }


    private void UpdateSettings(string setting)
    {
        if(setting == "all" || setting == "firstpersonreplay")
        {
            UpdateLayout();
        }
    }


    private void UpdateReplayMode(bool replayMode)
    {
        UpdateLayout();
    }


    private void OnEnable()
    {
        nonActiveCameraToggle.onClick.AddListener(ToggleNonActiveCameras);
        SettingsManager.OnSettingsUpdated += UpdateSettings;
        ReplayManager.OnReplayModeChanged += UpdateReplayMode;
        CameraUpdater.OnFreecamUpdated += UpdateLayout;

        UpdateLayout();
    }


    private void OnDisable()
    {
        nonActiveCameraToggle.onClick.RemoveListener(ToggleNonActiveCameras);
        SettingsManager.OnSettingsUpdated -= UpdateSettings;
        ReplayManager.OnReplayModeChanged -= UpdateReplayMode;
        CameraUpdater.OnFreecamUpdated -= UpdateLayout;
    }
}


[Serializable]
public struct CameraSettingsSection
{
    public CameraSettingsSectionType Type;
    public RectTransform Header;
    public List<RectTransform> Items;
}


public enum CameraSettingsSectionType
{
    FirstPerson,
    Replay,
    Preview,
    Free
}
