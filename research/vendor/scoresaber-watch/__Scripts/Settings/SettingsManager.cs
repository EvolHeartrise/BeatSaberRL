using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;
using UnityEngine;

public class SettingsManager : MonoBehaviour
{
    private static Settings _currentSettings;
    public static Settings CurrentSettings
    {
        get => _currentSettings;

        set
        {
            _currentSettings = value;

            // Using the Loaded parameter avoids potentially expensive null checking
            // every single time a setting is read
            Loaded = _currentSettings != null;
        }
    }

    private static Settings _overrides;
    public static Settings Overrides
    {
        get => _overrides;

        set
        {
            _overrides = value;
            CheckShouldUseOverrides();
        }
    }

    public static event Action<string> OnSettingsUpdated;
    public static event Action OnSettingsReset;

    public static bool Loaded { get; private set; }
    public static bool UseOverrides { get; private set; }

#if !UNITY_WEBGL || UNITY_EDITOR
    private const string settingsFile = "UserSettings.json";

    private const float autoSaveLength = 0.25f;
    private static float dirtyTime = 0f;
#endif

    private static readonly Color PreviousUIColor = new Color(0.07058824f, 0.40784314f, 0.6313726f);
    private static readonly Color PoisonedUIColor = new Color(0.67058825f, 0.5803922f, 0.04313726f);
    private const int MobileDefaultFrameCap = 75;
    private const int CompactViewportWidth = 700;
    private const int CompactScreenWidth = 1100;
#if UNITY_WEBGL && !UNITY_EDITOR
    private const string WebAudioVolumeScaleMigrationSetting = "webaudiovolumescale";
    private const int WebAudioVolumeScaleMigrationVersion = 1;
    private static readonly string[] WebAudioVolumeSettings = new string[]
    {
        "musicvolume",
        "hitsoundvolume",
        "chainvolume"
    };
#endif

#if UNITY_WEBGL && !UNITY_EDITOR
    [System.Runtime.InteropServices.DllImport("__Internal")]
    private static extern int GetArcViewerViewportWidth();

    [System.Runtime.InteropServices.DllImport("__Internal")]
    private static extern int GetArcViewerViewportHeight();
#endif

    [SerializeField] private List<SerializedOption<bool>> defaultBools;
    [SerializeField] private List<SerializedOption<int>> defaultInts;
    [SerializeField] private List<SerializedOption<float>> defaultFloats;
    [SerializeField] private List<SerializedOption<Color>> defaultColors;


#if !UNITY_WEBGL || UNITY_EDITOR
    private bool saving;


    private async Task WriteFileAsync(string text, string path)
    {
        await File.WriteAllTextAsync(path, text);
    }


    private IEnumerator SaveSettingsCoroutine()
    {
        saving = true;

        string filePath = Path.Combine(Application.persistentDataPath, settingsFile);
        //Need to use newtonsoft otherwise dictionaries don't serialize
        string json = JsonConvert.SerializeObject(CurrentSettings);

        Task writeTask = WriteFileAsync(json, filePath);
        yield return new WaitUntil(() => writeTask.IsCompleted);

        if(writeTask.Exception != null)
        {
            Debug.Log($"Failed to save settings with error: {writeTask.Exception.Message}, {writeTask.Exception.StackTrace}");
            ErrorHandler.Instance?.ShowPopup(ErrorType.Error, "Failed to save your settings!");
        }

        saving = false;
    }


    public void SaveSettings()
    {
        if(saving)
        {
            Debug.LogWarning("Trying to save settings when already saving!");
            return;
        }

        StartCoroutine(SaveSettingsCoroutine());
    }


    private void LoadSettings()
    {
        string filePath = Path.Combine(Application.persistentDataPath, settingsFile);
        
        if(!File.Exists(filePath))
        {
            Debug.Log("Settings file doesn't exist. Using defaults.");
            CurrentSettings = Settings.GetDefaultSettings();
            SaveSettings();

            OnSettingsUpdated?.Invoke("all");
            return;
        }

        try
        {
            string json = File.ReadAllText(filePath);
            CurrentSettings = JsonConvert.DeserializeObject<Settings>(json);
        }
        catch(Exception err)
        {
            Debug.LogWarning($"Failed to load settings with error: {err.Message}, {err.StackTrace}");
            CurrentSettings = Settings.GetDefaultSettings();

            SaveSettings();
        }

        if(ApplyDefaultMigrations())
        {
            SetDirty();
        }

        OnSettingsUpdated?.Invoke("all");
    }


    private static void SetDirty()
    {
        dirtyTime = autoSaveLength;
    }
#endif


    public static void CheckShouldUseOverrides()
    {
        bool hadOverrides = UseOverrides;
        UseOverrides = Overrides != null && GetBool("allowoverride", false);

        if(UseOverrides != hadOverrides)
        {
            //Overrides have been enabled/disabled, update settings
            OnSettingsUpdated?.Invoke("all");
        }
    }


    public static bool GetBool(string name, bool useOverride = true)
    {
        bool value;

        if(useOverride && UseOverrides && Overrides.Bools.TryGetValue(name, out value))
        {
            return value;
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        if(CurrentSettings.Bools.TryGetValue(name, out value))
        {
            return value;
        }

        //Use ints for bools since PlayerPrefs can't store them
        int defaultValue = 0;
        if(Settings.DefaultSettings.Bools.TryGetValue(name, out value))
        {
            defaultValue = value ? 1 : 0;
        }

        //Save the setting to memory so we can avoid expensive PlayerPrefs calls
        value = PlayerPrefs.GetInt(name, defaultValue) > 0;
        if(!CurrentSettings.Bools.TryAdd(name, value))
        {
            Debug.LogWarning($"Failed to save setting {name} to memory!");
        }

        return value;
#else
        if(!Loaded)
        {
            Debug.LogWarning($"Setting {name} was accessed before settings loaded!");
            return false;
        }

        if(CurrentSettings.Bools.TryGetValue(name, out value))
        {
            return value;
        }
        else if(Settings.DefaultSettings.Bools.TryGetValue(name, out value))
        {
            return value;
        }
        else return false;
#endif
    }


    public static int GetInt(string name, bool useOverride = true)
    {
        int value;

        if(useOverride && UseOverrides && Overrides.Ints.TryGetValue(name, out value))
        {
            return value;
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        if(CurrentSettings.Ints.TryGetValue(name, out value))
        {
            return value;
        }

        //Value hasn't been loaded yet
        int defaultValue = 0;
        if(Settings.DefaultSettings.Ints.TryGetValue(name, out value))
        {
            defaultValue = value;
        }

        //Save the setting to memory so we can avoid expensive PlayerPrefs calls
        value = PlayerPrefs.GetInt(name, defaultValue);
        if(!CurrentSettings.Ints.TryAdd(name, value))
        {
            Debug.LogWarning($"Failed to save setting {name} to memory!");
        }

        return value;
#else
        if(!Loaded)
        {
            Debug.LogWarning($"Setting {name} was accessed before settings loaded!");
            return 0;
        }

        if(CurrentSettings.Ints.TryGetValue(name, out value))
        {
            return value;
        }
        else if(Settings.DefaultSettings.Ints.TryGetValue(name, out value))
        {
            return value;
        }
        else return 0;
#endif
    }


    public static float GetFloat(string name, bool useOverride = true)
    {
        float value;

        if(useOverride && UseOverrides && Overrides.Floats.TryGetValue(name, out value))
        {
            return value;
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        if(CurrentSettings.Floats.TryGetValue(name, out value))
        {
            return value;
        }

        //Value hasn't been loaded yet
        float defaultValue = 0f;
        if(Settings.DefaultSettings.Floats.TryGetValue(name, out value))
        {
            defaultValue = value;
        }

        //Save the setting to memory so we can avoid expensive PlayerPrefs calls
        value = PlayerPrefs.GetFloat(name, defaultValue);
        if(!CurrentSettings.Floats.TryAdd(name, value))
        {
            Debug.LogWarning($"Failed to save setting {name} to memory!");
        }

        return value;
#else
        if(!Loaded)
        {
            Debug.LogWarning($"Setting {name} was accessed before settings loaded!");
            return 0;
        }

        if(CurrentSettings.Floats.TryGetValue(name, out value))
        {
            return value;
        }
        else if(Settings.DefaultSettings.Floats.TryGetValue(name, out value))
        {
            return value;
        }
        else return 0f;
#endif
    }


    public static Color GetColor(string name, bool useOverride = true)
    {
        float r = Mathf.Clamp01(GetFloat(name + ".r", useOverride));
        float g = Mathf.Clamp01(GetFloat(name + ".g", useOverride));
        float b = Mathf.Clamp01(GetFloat(name + ".b", useOverride));
        return new Color(r, g, b);
    }


    public static void SetRule(string name, bool value, bool notify = true)
    {
        Dictionary<string, bool> rules = CurrentSettings.Bools;
        if(rules.ContainsKey(name))
        {
            rules[name] = value;
        }
        else
        {
            rules.Add(name, value);
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        PlayerPrefs.SetInt(name, value ? 1 : 0);
#else
        SetDirty();
#endif

        if(UseOverrides && Overrides.Bools.ContainsKey(name))
        {
            //Allow the user to veto the override by changing the setting
            Overrides.Bools.Remove(name);
        }

        if(notify)
        {
            OnSettingsUpdated?.Invoke(name);
        }
    }


    public static void SetRule(string name, int value, bool notify = true)
    {
        Dictionary<string, int> rules = CurrentSettings.Ints;
        if(rules.ContainsKey(name))
        {
            rules[name] = value;
        }
        else
        {
            rules.Add(name, value);
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        PlayerPrefs.SetInt(name, value);
#else
        SetDirty();
#endif

        if(UseOverrides && Overrides.Ints.ContainsKey(name))
        {
            //Allow the user to veto the override by changing the setting
            Overrides.Ints.Remove(name);
        }

        if(notify)
        {
            OnSettingsUpdated?.Invoke(name);
        }
    }


    public static void SetRule(string name, float value, bool notify = true, bool round = true)
    {
        if(round)
        {
            value = (float)Math.Round(value, 3);
        }

        Dictionary<string, float> rules = CurrentSettings.Floats;
        if(rules.ContainsKey(name))
        {
            rules[name] = value;
        }
        else
        {
            rules.Add(name, value);
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        PlayerPrefs.SetFloat(name, value);
#else
        SetDirty();
#endif

        if(UseOverrides && Overrides.Floats.ContainsKey(name))
        {
            //Allow the user to veto the override by changing the setting
            Overrides.Floats.Remove(name);
        }

        if(notify)
        {
            OnSettingsUpdated?.Invoke(name);
        }
    }


    public static void SetRule(string name, Color value, bool notify = true)
    {
        //Color values need to be set as separate floats
        SetRule(name + ".r", value.r, false, false);
        SetRule(name + ".g", value.g, false, false);
        SetRule(name + ".b", value.b, false, false);

        if(notify)
        {
            OnSettingsUpdated?.Invoke(name);
        }
    }


    public static void SetDefaults()
    {
        bool staticLightsHintDismissed = GetBool("staticlightshintdismissed");
        bool replayMode = GetBool("replaymode");

#if UNITY_WEBGL && !UNITY_EDITOR
        PlayerPrefs.DeleteAll();
        PlayerPrefs.SetInt(WebAudioVolumeScaleMigrationSetting, WebAudioVolumeScaleMigrationVersion);
        CurrentSettings = new Settings();
#else
        CurrentSettings = Settings.GetDefaultSettings();
        SetDirty();
#endif

        //Also clear overrides
        Overrides = null;
        
        //Some settings should still persist or else they'll be annoying
        SetRule("staticlightshintdismissed", staticLightsHintDismissed, false);
        SetRule("replaymode", replayMode, false);

        OnSettingsReset?.Invoke();
        OnSettingsUpdated?.Invoke("all");

        ErrorHandler.Instance.ShowPopup(ErrorType.Notification, "Settings have been reset.");
    }


    private static bool ApplyDefaultMigrations()
    {
        bool changed = false;

        changed |= MigrateBoolDefault("firstpersonreplay", false, true);
        changed |= MigrateFirstPersonCameraDefaults();
        changed |= RestorePoisonedUIColorDefault();
        changed |= MigrateBoolDefault("showheadset", false, true);
        changed |= MigrateBoolDefault("forcefpcameraupright", false, true);

#if UNITY_WEBGL && !UNITY_EDITOR
        changed |= MigrateWebAudioVolumeScale();

        if(changed)
        {
            PlayerPrefs.Save();
        }
#endif

        return changed;
    }

#if UNITY_WEBGL && !UNITY_EDITOR
    private static bool MigrateWebAudioVolumeScale()
    {
        if(PlayerPrefs.GetInt(WebAudioVolumeScaleMigrationSetting, 0) >= WebAudioVolumeScaleMigrationVersion)
        {
            return false;
        }

        foreach(string setting in WebAudioVolumeSettings)
        {
            if(!PlayerPrefs.HasKey(setting))
            {
                continue;
            }

            float oldGain = Mathf.Clamp01(PlayerPrefs.GetFloat(setting));
            SetMigratedFloat(setting, Mathf.Sqrt(oldGain));
        }

        PlayerPrefs.SetInt(WebAudioVolumeScaleMigrationSetting, WebAudioVolumeScaleMigrationVersion);
        return true;
    }
#endif


    private static bool MigrateFirstPersonCameraDefaults()
    {
        bool hasStoredCameraSettings =
            HasStoredInt("fpcamerafov") ||
            HasStoredFloat("fpcameraposition") ||
            HasStoredFloat("fpcameramovementsmoothing") ||
            HasStoredFloat("fpcamerarotationsmoothing") ||
            HasStoredInt("fpcamerarotoffset");

        bool usesOldCameraSettings =
            IntUsesOldDefault("fpcamerafov", 75) &&
            FloatUsesOldDefault("fpcameraposition", 1f) &&
            FloatUsesOldDefault("fpcameramovementsmoothing", 0f) &&
            FloatUsesOldDefault("fpcamerarotationsmoothing", 0.3f) &&
            IntUsesOldDefault("fpcamerarotoffset", 0);

        if(!hasStoredCameraSettings || !usesOldCameraSettings)
        {
            return false;
        }

        SetMigratedInt("fpcamerafov", 70);
        SetMigratedFloat("fpcameraposition", 0.7f);
        SetMigratedFloat("fpcameramovementsmoothing", 0f);
        SetMigratedFloat("fpcamerarotationsmoothing", 0.3f);
        SetMigratedInt("fpcamerarotoffset", -15);
        return true;
    }


    private static bool RestorePoisonedUIColorDefault()
    {
        bool hasStoredPoisonedUIColor =
            HasStoredBool("useuicolor") &&
            HasStoredFloat("uicolor.r") &&
            HasStoredFloat("uicolor.g") &&
            HasStoredFloat("uicolor.b");

        bool usesPoisonedUIColor =
            BoolUsesOldDefault("useuicolor", true) &&
            FloatUsesOldDefault("uicolor.r", PoisonedUIColor.r) &&
            FloatUsesOldDefault("uicolor.g", PoisonedUIColor.g) &&
            FloatUsesOldDefault("uicolor.b", PoisonedUIColor.b);

        if(!hasStoredPoisonedUIColor || !usesPoisonedUIColor)
        {
            return false;
        }

        SetMigratedBool("useuicolor", false);
        SetMigratedFloat("uicolor.r", PreviousUIColor.r);
        SetMigratedFloat("uicolor.g", PreviousUIColor.g);
        SetMigratedFloat("uicolor.b", PreviousUIColor.b);
        return true;
    }


    private static bool MigrateBoolDefault(string name, bool oldValue, bool newValue)
    {
        if(!HasStoredBool(name) || !BoolUsesOldDefault(name, oldValue))
        {
            return false;
        }

        SetMigratedBool(name, newValue);
        return true;
    }


    private static bool MigrateColorDefault(string name, Color oldValue, Color newValue)
    {
        string red = name + ".r";
        string green = name + ".g";
        string blue = name + ".b";

        bool hasStoredColor = HasStoredFloat(red) || HasStoredFloat(green) || HasStoredFloat(blue);
        bool usesOldColor =
            FloatUsesOldDefault(red, oldValue.r) &&
            FloatUsesOldDefault(green, oldValue.g) &&
            FloatUsesOldDefault(blue, oldValue.b);

        if(!hasStoredColor || !usesOldColor)
        {
            return false;
        }

        SetMigratedFloat(red, newValue.r);
        SetMigratedFloat(green, newValue.g);
        SetMigratedFloat(blue, newValue.b);
        return true;
    }


    private static bool HasStoredBool(string name)
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        return PlayerPrefs.HasKey(name);
#else
        return CurrentSettings.Bools.ContainsKey(name);
#endif
    }


    private static bool HasStoredInt(string name)
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        return PlayerPrefs.HasKey(name);
#else
        return CurrentSettings.Ints.ContainsKey(name);
#endif
    }


    private static bool HasStoredFloat(string name)
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        return PlayerPrefs.HasKey(name);
#else
        return CurrentSettings.Floats.ContainsKey(name);
#endif
    }


    private static bool BoolUsesOldDefault(string name, bool oldValue)
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        return !PlayerPrefs.HasKey(name) || (PlayerPrefs.GetInt(name) > 0) == oldValue;
#else
        return !CurrentSettings.Bools.TryGetValue(name, out bool value) || value == oldValue;
#endif
    }


    private static bool IntUsesOldDefault(string name, int oldValue)
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        return !PlayerPrefs.HasKey(name) || PlayerPrefs.GetInt(name) == oldValue;
#else
        return !CurrentSettings.Ints.TryGetValue(name, out int value) || value == oldValue;
#endif
    }


    private static bool FloatUsesOldDefault(string name, float oldValue)
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        return !PlayerPrefs.HasKey(name) || Mathf.Approximately(PlayerPrefs.GetFloat(name), oldValue);
#else
        return !CurrentSettings.Floats.TryGetValue(name, out float value) || Mathf.Approximately(value, oldValue);
#endif
    }


    private static void SetMigratedBool(string name, bool value)
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        PlayerPrefs.SetInt(name, value ? 1 : 0);
#endif
        CurrentSettings.Bools[name] = value;
    }


    private static void SetMigratedInt(string name, int value)
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        PlayerPrefs.SetInt(name, value);
#endif
        CurrentSettings.Ints[name] = value;
    }


    private static void SetMigratedFloat(string name, float value)
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        PlayerPrefs.SetFloat(name, value);
#endif
        CurrentSettings.Floats[name] = value;
    }


    private void UpdateUIState(UIState newState)
    {
        if(UseOverrides && newState == UIState.MapSelection)
        {
            //Clear settings overrides when exiting a map
            Overrides = null;
            OnSettingsUpdated?.Invoke("all");
        }
    }


    private static void ApplyRuntimeDefaults()
    {
        bool mobileViewport = IsMobileViewport();

        Settings.DefaultSettings.Bools[GraphicSettingsUpdater.CapFpsSetting] = true;
        Settings.DefaultSettings.Bools[GraphicSettingsUpdater.MatchRefreshSetting] = !mobileViewport;
        Settings.DefaultSettings.Ints[GraphicSettingsUpdater.FpsLimitSetting] =
            mobileViewport ? MobileDefaultFrameCap : GetDisplayRefreshRate();
    }


    private static bool IsMobileViewport()
    {
        if(Application.isMobilePlatform)
        {
            return true;
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        int viewportWidth = Mathf.Max(1, GetArcViewerViewportWidth());
        int viewportHeight = Mathf.Max(1, GetArcViewerViewportHeight());
        return viewportHeight > viewportWidth * 1.15f
            || viewportWidth <= CompactViewportWidth
            || (Screen.height > Screen.width * 1.15f && Screen.width <= CompactScreenWidth);
#else
        return false;
#endif
    }


    public static int GetDisplayRefreshRate()
    {
        int refreshRate = Mathf.RoundToInt((float)Screen.currentResolution.refreshRateRatio.value);
        return refreshRate > 0 ? Mathf.Clamp(refreshRate, 1, 999) : MobileDefaultFrameCap;
    }


    private void Awake()
    {
        UIStateManager.OnUIStateChanged += UpdateUIState;

        //Update the default settings
        Settings.DefaultSettings.Bools = Settings.SerializedOptionsToDictionary<bool>(defaultBools);
        Settings.DefaultSettings.Ints = Settings.SerializedOptionsToDictionary<int>(defaultInts);
        Settings.DefaultSettings.Floats = Settings.SerializedOptionsToDictionary<float>(defaultFloats);
        Settings.DefaultSettings.AddColorRules(defaultColors);
        ApplyRuntimeDefaults();

#if !UNITY_WEBGL || UNITY_EDITOR
        //Load settings from json if not running in WebGL
        //Otherwise settings are handled through playerprefs instead
        LoadSettings();
#else
        CurrentSettings = new Settings();
        ApplyDefaultMigrations();
        OnSettingsUpdated?.Invoke("all");
#endif
    }


#if !UNITY_WEBGL || UNITY_EDITOR
    private void Update()
    {
        if(dirtyTime > 0f)
        {
            //A setting has been changed, count down the timer to save settings
            dirtyTime -= Time.deltaTime;

            if(dirtyTime <= 0f)
            {
                SaveSettings();
            }
        }
    }
#endif
}


[Serializable]
public class Settings
{
    public Dictionary<string, bool> Bools;
    public Dictionary<string, int> Ints;
    public Dictionary<string, float> Floats;


    public Settings()
    {
        Bools = new Dictionary<string, bool>();
        Ints = new Dictionary<string, int>();
        Floats = new Dictionary<string, float>();
    }


    public void AddColorRule(string name, Color color)
    {
        bool success = Floats.TryAdd(name + ".r", color.r);
        success &= Floats.TryAdd(name + ".g", color.g);
        success &= Floats.TryAdd(name + ".b", color.b);
        if(!success)
        {
            Debug.LogWarning($"Failed to add setting '{name}'. Is it a duplicate?");
        }
    }


    public void AddColorRules(IEnumerable<SerializedOption<Color>> colors)
    {
        foreach(SerializedOption<Color> color in colors)
        {
            AddColorRule(color.Name, color.Value);
        }
    }


    public static Settings DefaultSettings = new Settings();
    public static Settings GetDefaultSettings()
    {
        //Provides a deep copy of the default settings I hate reference types I hate reference types I hate reference types I hate reference types I hate reference types
        Settings settings = new Settings
        {
            Bools = new Dictionary<string, bool>(),
            Ints = new Dictionary<string, int>(),
            Floats = new Dictionary<string, float>()
        };

        foreach(var key in DefaultSettings.Bools)
        {
            settings.Bools.Add(key.Key, key.Value);
        }

        foreach(var key in DefaultSettings.Ints)
        {
            settings.Ints.Add(key.Key, key.Value);
        }

        foreach(var key in DefaultSettings.Floats)
        {
            settings.Floats.Add(key.Key, key.Value);
        }

        return settings;
    }


    public static Dictionary<string, T> SerializedOptionsToDictionary<T>(List<SerializedOption<T>> options)
    {
        Dictionary<string, T> dictionary = new Dictionary<string, T>();

        foreach(SerializedOption<T> option in options)
        {
#if UNITY_WEBGL
            T value = option.ValueWebGL.Enabled ? option.ValueWebGL.Value : option.Value;
            bool success = dictionary.TryAdd(option.Name, value);
#else
            bool success = dictionary.TryAdd(option.Name, option.Value);
#endif
            if(!success)
            {
                Debug.LogWarning($"Failed to add setting '{option.Name}'. Is it a duplicate?");
            }
        }

        return dictionary;
    }


    public Settings ExportOverrides()
    {
        Settings overrides = new Settings();

        foreach(KeyValuePair<string, bool> pair in Bools)
        {
            //Compare against the default value
            bool defaultValue;
            if(!DefaultSettings.Bools.TryGetValue(pair.Key, out defaultValue))
            {
                defaultValue = false;
            }

            if(pair.Value != defaultValue)
            {
                //This setting is different from default, add it to the overrides
                overrides.Bools.Add(pair.Key, pair.Value);
            }
        }

        foreach(KeyValuePair<string, int> pair in Ints)
        {
            //Compare against the default value
            int defaultValue;
            if(!DefaultSettings.Ints.TryGetValue(pair.Key, out defaultValue))
            {
                defaultValue = 0;
            }

            if(pair.Value != defaultValue)
            {
                //This setting is different from default, add it to the overrides
                overrides.Ints.Add(pair.Key, pair.Value);
            }
        }

        foreach(KeyValuePair<string, float> pair in Floats)
        {
            //Compare against the default value
            float defaultValue;
            if(!DefaultSettings.Floats.TryGetValue(pair.Key, out defaultValue))
            {
                defaultValue = 0f;
            }

            if(pair.Value != defaultValue)
            {
                //This setting is different from default, add it to the overrides
                overrides.Floats.Add(pair.Key, pair.Value);
            }
        }

        return overrides;
    }
}


[Serializable]
public struct SerializedOption<T>
{
    public string Name;
    public T Value;
    public Optional<T> ValueWebGL;
}