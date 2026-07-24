using System;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class SettingsSlider : MonoBehaviour
{
    [Header("Components")]
    [SerializeField] private Slider slider;
    [SerializeField] private TMP_InputField valueInput;
    [SerializeField] private RectTransform valueText;
    [SerializeField] private TextMeshProUGUI nameLabel;

    [Header("Configuration")]
    [SerializeField] private string minOverride;
    [SerializeField] private string maxOverride;
    [SerializeField] private string rule;

    [Space]
    [SerializeField] private bool integerValue;
    [SerializeField] private Optional<float> stepAmount;
    [SerializeField] private float minValue = 0f;
    [SerializeField] private float maxValue = 1f;

    [Space]
    [SerializeField] private bool hideInWebGL;
    [SerializeField] private bool realTimeUpdates = true;
    [SerializeField] private Optional<SerializedOption<bool>> requiredSetting;

    [Space]
    [SerializeField] private Color enabledColor;
    [SerializeField] private Color disabledColor;

    private SliderPointerUpHandler pointerUpHandler;


    public void SetValue(float value)
    {
        UpdateValue(GetSliderValue());
    }


    public void SetValueText(float value)
    {
        UpdateText(GetSliderValue());
    }


    public void SetValue(string value)
    {
        if(float.TryParse(value, out float number))
        {
            number = ClampValue(number);

            if(integerValue)
            {
                number = Mathf.RoundToInt(number);
            }
            SetSliderValue(number);

            UpdateValue(number);
            UpdateText(number);
        }
        else UpdateText(GetSliderValue());

        //Force de-select the text field
        EventSystemHelper.SetSelectedGameObject(null);
    }


    private void UpdateSettings(string changedSetting)
    {
        if(changedSetting == "all" || changedSetting == rule)
        {
            float newValue = integerValue ? SettingsManager.GetInt(rule, false) : SettingsManager.GetFloat(rule, false);
            float clampedValue = ClampValue(newValue);
            if(!Mathf.Approximately(newValue, clampedValue))
            {
                UpdateValue(clampedValue, false);
            }

            SetSliderValue(clampedValue);
            UpdateText(clampedValue);
        }

        if(requiredSetting.Enabled)
        {
            CheckRequiredSetting(changedSetting);
        }
    }


    private void CheckRequiredSetting(string changedSetting)
    {
        if(rule == GraphicSettingsUpdater.FpsLimitSetting)
        {
            if(changedSetting == "all"
                || changedSetting == GraphicSettingsUpdater.CapFpsSetting
                || changedSetting == GraphicSettingsUpdater.MatchRefreshSetting)
            {
                SetInteractable(SettingsManager.GetBool(GraphicSettingsUpdater.CapFpsSetting, false)
                    && !SettingsManager.GetBool(GraphicSettingsUpdater.MatchRefreshSetting, false));
            }

            return;
        }

        SerializedOption<bool> option = requiredSetting.Value;
        if(changedSetting == "all" || changedSetting == option.Name)
        {
            SetInteractable(option.Value == SettingsManager.GetBool(option.Name, false));
        }
    }


    private void SetInteractable(bool interactable)
    {
        slider.interactable = interactable;
        valueInput.interactable = interactable;
        nameLabel.color = interactable ? enabledColor : disabledColor;
    }
    

    private void UpdateValue(float value, bool notify = true)
    {
        value = ClampValue(value);
        if(integerValue)
        {
            SettingsManager.SetRule(rule, Mathf.RoundToInt(value), notify);
        }
        else SettingsManager.SetRule(rule, value, notify);
    }


    private void UpdateText(float value)
    {
        value = ClampValue(value);
        float effectiveMaxValue = GetEffectiveMaxValue();

        if(value > effectiveMaxValue - 0.005 && maxOverride != "")
        {
            valueInput.SetTextWithoutNotify(maxOverride);
        }
        else if(value < minValue + 0.005 && minOverride != "")
        {
            valueInput.SetTextWithoutNotify(minOverride);
        }
        else valueInput.SetTextWithoutNotify(Math.Round(value, 2).ToString());

        valueText.anchoredPosition = Vector2.zero;
    }


    private void SetSliderValue(float value)
    {
        value = ClampValue(value);
        if(!stepAmount.Enabled)
        {
            slider.SetValueWithoutNotify(value);
            return;
        }

        float convertedValue = (value - minValue) / stepAmount.Value;
        slider.SetValueWithoutNotify(convertedValue);
    }


    private float GetSliderValue()
    {
        if(!stepAmount.Enabled)
        {
            return ClampValue(slider.value);
        }

        float sliderValue = (slider.value * stepAmount.Value) + minValue;
        return ClampValue(sliderValue);
    }


    private float ClampValue(float value)
    {
        return Mathf.Clamp(value, minValue, GetEffectiveMaxValue());
    }


    private float GetEffectiveMaxValue()
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        if(rule == GraphicSettingsUpdater.FpsLimitSetting)
        {
            return Mathf.Min(maxValue, SettingsManager.GetDisplayRefreshRate());
        }
#endif

        return maxValue;
    }


    private void OnEnable()
    {
#if UNITY_WEBGL
        if(hideInWebGL)
        {
            gameObject.SetActive(false);
            return;
        }
#endif

        if(!pointerUpHandler)
        {
            pointerUpHandler = slider.GetComponent<SliderPointerUpHandler>();
        }

        slider.wholeNumbers = integerValue || stepAmount.Enabled;
        if(stepAmount.Enabled)
        {
            //Turn the slider into an integer slider, and convert the min and max
            //into an equivalent number of steps
            float valueRange = GetEffectiveMaxValue() - minValue;
            int numSteps = (int)(valueRange / stepAmount.Value);

            slider.minValue = 0;
            slider.maxValue = numSteps;
        }
        else
        {
            slider.minValue = minValue;
            slider.maxValue = GetEffectiveMaxValue();
        }

        if(integerValue)
        {
            int newValue = SettingsManager.GetInt(rule);

            SetSliderValue(newValue);
            UpdateText(newValue);
        }
        else
        {
            float newValue = SettingsManager.GetFloat(rule);

            SetSliderValue(newValue);
            UpdateText(newValue);
        }

        slider.onValueChanged.AddListener(SetValueText);

        if(realTimeUpdates)
        {
            slider.onValueChanged.AddListener(SetValue);
        }
        else
        {
            pointerUpHandler.OnSliderEnd.AddListener(SetValue);
        }

        SettingsManager.OnSettingsUpdated += UpdateSettings;
        UpdateSettings("all");
    }


    private void OnDisable()
    {
        if(slider)
        {
            slider.onValueChanged.RemoveAllListeners();
        }
        SettingsManager.OnSettingsUpdated -= UpdateSettings;
    }
}