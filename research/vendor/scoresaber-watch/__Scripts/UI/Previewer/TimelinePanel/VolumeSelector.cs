using System;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using TMPro;
using System.Linq;

public class VolumeSelector : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler
{
    [SerializeField] private GameObject sliderContainer;

    [Space]
    [SerializeField] private Slider musicSlider;
    [SerializeField] private TMP_InputField musicInputField;
    [SerializeField] private Slider hitsoundSlider;
    [SerializeField] private TMP_InputField hitsoundInputField;

    [Space]
    [SerializeField] private Color defaultColor;
    [SerializeField] private Color highlightedColor;

    [Space]
    [SerializeField] private Image image;
    [SerializeField] private Tooltip tooltip;
    [SerializeField] private Sprite activeSprite;
    [SerializeField] private Sprite mutedSprite;
    [SerializeField] private string activeTooltip;
    [SerializeField] private string mutedTooltip;

    private bool hovered;
    private bool clicked;
    private bool musicTextSelected;
    private bool hitsoundTextSelected;

    private const float sliderStepScale = 0.05f;
    private const string musicRule = "musicvolume";
    private const string hitsoundRule = "hitsoundvolume";
    private const string musicEnabledRule = "enablemusic";
    private const string hitsoundEnabledRule = "enablehitsound";


    public void ShowSliders()
    {
        image.color = highlightedColor;

        sliderContainer.SetActive(true);
        UpdateValueDisplay();
    }


    public void HideSliders()
    {
        //Don't hide the slider if the mouse is over us, or if we're still clicked on
        if(hovered || clicked || musicTextSelected || hitsoundTextSelected) return;

        image.color = defaultColor;

        sliderContainer.SetActive(false);
    }


    public void OnMuteButtonClick()
    {
        bool muted = IsMuted();

        SetMusicMuted(!muted);
        SetHitsoundMuted(!muted);

        UpdateSprite();
    }


    public void OnPointerEnter(PointerEventData eventData)
    {
        hovered = true;
        ShowSliders();
    }


    public void OnPointerExit(PointerEventData eventData)
    {
        hovered = false;
        HideSliders();
    }


    public void OnMusicTextSelected()
    {
        musicTextSelected = true;
        ShowSliders();
    }


    public void OnMusicTextDeselected()
    {
        musicTextSelected = false;
        HideSliders();
    }


    public void OnHitsoundTextSelected()
    {
        hitsoundTextSelected = true;
        ShowSliders();
    }


    public void OnHitsoundTextDeselected()
    {
        hitsoundTextSelected = false;
        HideSliders();
    }


    public void SetMusicVolumeSlider(float value)
    {
        SetMusicVolume(value * sliderStepScale);
        UpdateValueDisplay();
    }


    public void SetHitsoundVolumeSlider(float value)
    {
        SetHitsoundVolume(value * sliderStepScale);
        UpdateValueDisplay();
    }


    public void SetMusicVolume(string value)
    {
        if(float.TryParse(value, out float newVolume))
        {
            SetMusicVolume(newVolume);
        }

        //Force deselect the input field
        EventSystemHelper.SetSelectedGameObject(null);
        UpdateValueDisplay();
    }


    public void SetHitsoundVolume(string value)
    {
        if(float.TryParse(value, out float newVolume))
        {
            SetHitsoundVolume(newVolume);
        }

        //Force deselect the input field
        EventSystemHelper.SetSelectedGameObject(null);
        UpdateValueDisplay();
    }

    private void SetMusicMuted(bool muted)
    {
        SettingsManager.SetRule(musicEnabledRule, !muted);
    }


    private void SetHitsoundMuted(bool muted)
    {
        SettingsManager.SetRule(hitsoundEnabledRule, !muted);
    }


    private void SetMusicVolume(float musicVolume)
    {
        SetMusicMuted(musicVolume == 0f);
        musicVolume = (float)Math.Round(musicVolume, 2);
        SettingsManager.SetRule(musicRule, musicVolume);
    }


    private void SetHitsoundVolume(float hitsoundVolume)
    {
        SetHitsoundMuted(hitsoundVolume == 0f);
        hitsoundVolume = (float)Math.Round(hitsoundVolume, 2);
        SettingsManager.SetRule(hitsoundRule, hitsoundVolume);
    }


    private bool IsMusicMuted()
    {
        return !SettingsManager.GetBool(musicEnabledRule);
    }

    
    private bool IsHitsoundMuted()
    {
        return !SettingsManager.GetBool(hitsoundEnabledRule);
    }


    private bool IsMuted()
    {
        return IsMusicMuted() && IsHitsoundMuted();
    }


    private float GetMusicVolume()
    {
        float musicVolume = Mathf.Clamp01(SettingsManager.GetFloat(musicRule));
        return IsMusicMuted() ? 0 : (float)Math.Round(musicVolume, 2);
    }


    private float GetHitsoundVolume()
    {
        float hitsoundVolume = Mathf.Clamp01(SettingsManager.GetFloat(hitsoundRule));
        return IsHitsoundMuted() ? 0 : (float)Math.Round(hitsoundVolume, 2);
    }


    private void UpdateSprite()
    {
        bool muted = IsMuted();
        image.sprite = muted ? mutedSprite : activeSprite;
        tooltip.Text = muted ? mutedTooltip : activeTooltip;
        tooltip.ForceUpdate();
    }


    private void UpdateValueDisplay()
    {
        float musicVolume = GetMusicVolume();
        float hitsoundVolume = GetHitsoundVolume();

        musicSlider.SetValueWithoutNotify(musicVolume / sliderStepScale);
        musicInputField.SetTextWithoutNotify(musicVolume <= 0 ? "Off" : musicVolume.ToString());

        hitsoundSlider.SetValueWithoutNotify(hitsoundVolume / sliderStepScale);
        hitsoundInputField.SetTextWithoutNotify(hitsoundVolume <= 0 ? "Off" : hitsoundVolume.ToString());

        UpdateSprite();
    }


    private void UpdateSettings(string setting)
    {
        if(!sliderContainer.activeInHierarchy)
        {
            return;
        }

        if(setting == "all" || (new string[] { hitsoundRule, musicRule, hitsoundEnabledRule, musicEnabledRule }).Contains(setting))
        {
            UpdateValueDisplay();
        }
    }


    private void Update()
    {
        if(hovered && Input.GetMouseButtonDown(0))
        {
            //Weird hack since IPointerDownHandler doesn't include children in raycast for some reason
            clicked = true;
        }
        else if(clicked && !Input.GetMouseButton(0))
        {
            clicked = false;
            HideSliders();
        }
    }


    private void OnEnable()
    {
        SettingsManager.OnSettingsUpdated += UpdateSettings;
        HideSliders();
        UpdateSprite();
    }


    private void OnDisable()
    {
        SettingsManager.OnSettingsUpdated -= UpdateSettings;
    }
}