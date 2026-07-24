using System.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class GraphicSettingsUpdater : MonoBehaviour
{
    public const string CapFpsSetting = "capfps";
    public const string MatchRefreshSetting = "vsync";
    public const string FpsLimitSetting = "framecap";

    private const int IdleFrameRate = 30;
    private const float IdleFrameRateDelay = 1.5f;

    [SerializeField] private Volume bloomVolume;
    [SerializeField] private UniversalRenderPipelineAsset urpAsset;
    [SerializeField] private RenderTexture orthoCameraTexture;
    [SerializeField] private Camera lightGlowCamera;

    [Space]
    [SerializeField] private float defaultBloomStrength;

    private Bloom bloom;
    private Coroutine targetFrameRateRefreshCoroutine;
    private Vector3 lastInputMousePosition;
    private float activeFrameRateUntil;
    private bool usingIdleFrameRate;


    private void SetOrthoCameraMSAA(int msaa)
    {
#if !UNITY_WEBGL && !UNITY_EDITOR
        orthoCameraTexture.Release();
        orthoCameraTexture.antiAliasing = msaa;
        orthoCameraTexture.Create();
#endif
    }


    private int GetMSAA(int antiAliasing)
    {
        switch(antiAliasing)
        {
            default:
            case 0:
                return 1;
            case 1:
                return 2;
            case 2:
                return 4;
            case 3:
                return 8;
        }
    }


    private Camera GetLightGlowCamera()
    {
        if(lightGlowCamera)
        {
            return lightGlowCamera;
        }

        Camera mainCamera = Camera.main;
        Transform lightGlowCameraTransform = mainCamera ? mainCamera.transform.Find("Bloomfog/LightGlowCamera") : null;
        if(lightGlowCameraTransform)
        {
            lightGlowCamera = lightGlowCameraTransform.GetComponent<Camera>();
        }

        return lightGlowCamera;
    }


    private void SetLightGlowCameraEnabled(bool enabled)
    {
        Camera camera = GetLightGlowCamera();
        if(camera)
        {
            camera.enabled = enabled;
        }
    }


    private int GetRefreshFrameRate(int frameRate)
    {
        return frameRate == 999 ? 998 : frameRate + 1;
    }


    private IEnumerator RefreshTargetFrameRateNextFrame(int frameRate)
    {
        yield return null;
        Application.targetFrameRate = frameRate;
        targetFrameRateRefreshCoroutine = null;
    }


    private void CancelTargetFrameRateRefresh()
    {
        if(targetFrameRateRefreshCoroutine == null)
        {
            return;
        }

        StopCoroutine(targetFrameRateRefreshCoroutine);
        targetFrameRateRefreshCoroutine = null;
    }


    private void ApplyTargetFrameRate(int frameRate, bool forceRefresh)
    {
        CancelTargetFrameRateRefresh();

        if(forceRefresh && isActiveAndEnabled)
        {
            Application.targetFrameRate = GetRefreshFrameRate(frameRate);
            targetFrameRateRefreshCoroutine = StartCoroutine(RefreshTargetFrameRateNextFrame(frameRate));
            return;
        }

        Application.targetFrameRate = frameRate;
    }


    private int GetConfiguredFrameRate()
    {
        return Mathf.Clamp(SettingsManager.GetInt(FpsLimitSetting, false), 1, GetMaxConfiguredFrameRate());
    }


    private int GetMaxConfiguredFrameRate()
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        return SettingsManager.GetDisplayRefreshRate();
#else
        return 999;
#endif
    }


    private int GetIdleFrameRate(int configuredFrameRate)
    {
        return Mathf.Clamp(IdleFrameRate, 1, configuredFrameRate);
    }


    private bool ShouldUseIdleFrameRate(bool capFps)
    {
        return capFps
            && Time.unscaledTime >= activeFrameRateUntil
            && !TimeManager.Playing
            && !TimeManager.Scrubbing
            && !MapLoader.Loading
            && !HotReloader.Loading
            && !EnvironmentManager.Loading;
    }


    private bool HasUserInput()
    {
        Vector3 mousePosition = Input.mousePosition;
        bool mouseMoved = mousePosition != lastInputMousePosition;
        lastInputMousePosition = mousePosition;

        return mouseMoved
            || Input.anyKey
            || Input.mouseScrollDelta.sqrMagnitude > 0f
            || Input.touchCount > 0;
    }


    private void KeepActiveFrameRate()
    {
        activeFrameRateUntil = Time.unscaledTime + IdleFrameRateDelay;
    }


    private void UpdateAdaptiveFrameRate()
    {
        if(!SettingsManager.Loaded)
        {
            return;
        }

        if(HasUserInput() || TimeManager.Playing || TimeManager.Scrubbing)
        {
            KeepActiveFrameRate();
        }

        bool shouldUseIdleFrameRate = ShouldUseIdleFrameRate(SettingsManager.GetBool(CapFpsSetting, false));
        if(shouldUseIdleFrameRate != usingIdleFrameRate)
        {
            ApplyFrameLimiter(false);
        }
    }


    private void ApplyFrameLimiter(bool forceTargetFrameRateRefresh)
    {
        bool capFps = SettingsManager.GetBool(CapFpsSetting, false);
        bool matchRefresh = SettingsManager.GetBool(MatchRefreshSetting, false);
        int frameRate = GetConfiguredFrameRate();
        usingIdleFrameRate = ShouldUseIdleFrameRate(capFps);

        if(!capFps)
        {
            CancelTargetFrameRateRefresh();
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = -1;
        }
        else if(usingIdleFrameRate)
        {
            QualitySettings.vSyncCount = 0;
            ApplyTargetFrameRate(GetIdleFrameRate(frameRate), forceTargetFrameRateRefresh);
        }
        else if(matchRefresh)
        {
#if UNITY_WEBGL && !UNITY_EDITOR
            CancelTargetFrameRateRefresh();
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = -1;
#else
            CancelTargetFrameRateRefresh();
            QualitySettings.vSyncCount = 1;
            Application.targetFrameRate = -1;
#endif
        }
        else
        {
            QualitySettings.vSyncCount = 0;
            ApplyTargetFrameRate(frameRate, forceTargetFrameRateRefresh);
        }
    }


    public void UpdateGraphicsSettings(string setting)
    {
        bool allSettings = setting == "all";

        if(allSettings || setting == CapFpsSetting || setting == MatchRefreshSetting || setting == FpsLimitSetting)
        {
            KeepActiveFrameRate();
            ApplyFrameLimiter(allSettings || setting == CapFpsSetting || setting == MatchRefreshSetting);
        }

#if !UNITY_WEBGL || UNITY_EDITOR
        if(allSettings || setting == "antialiasing")
        {
            int antiAliasing = SettingsManager.GetInt("antialiasing", false);
            Camera.main.allowMSAA = antiAliasing > 0;

            int msaa = GetMSAA(antiAliasing);
            urpAsset.msaaSampleCount = msaa;
            SetOrthoCameraMSAA(msaa);
        }
#else
        if(allSettings)
        {
            Camera.main.allowMSAA = false;
        }
#endif

        if(allSettings || setting == "bloom")
        {
            bloom.intensity.value = defaultBloomStrength * Mathf.Clamp(SettingsManager.GetFloat("bloom"), 0f, 2f);
            bloom.active = bloom.intensity.value >= 0.001f;
        }

        if(allSettings || setting == "renderscale")
        {
            urpAsset.renderScale = Mathf.Clamp(SettingsManager.GetFloat("renderscale", false), 0.5f, 2f);
        }

        if(allSettings || setting == "upscaling")
        {
            bool useUpscaling = SettingsManager.GetBool("upscaling", false);
            urpAsset.upscalingFilter = useUpscaling ? UpscalingFilterSelection.FSR : UpscalingFilterSelection.Auto;
        }

        if(allSettings || setting == "bloomfogquality" || setting == "lightglowbrightness")
        {
            float bloomfogBrightness = Mathf.Clamp(SettingsManager.GetFloat("lightglowbrightness"), 0f, 2f);
            bool bloomfogEnabled = bloomfogBrightness >= 0.001f;

            Bloomfog.Enabled = bloomfogEnabled;
            SetLightGlowCameraEnabled(bloomfogEnabled);

            if(bloomfogEnabled)
            {
                Bloomfog.Quality = SettingsManager.GetInt("bloomfogquality", false);
            }
        }
    }


    private void UpdatePlaying(bool playing)
    {
        if(!playing || !SettingsManager.Loaded)
        {
            return;
        }

        KeepActiveFrameRate();
        ApplyFrameLimiter(false);
    }


    private void UpdateLiveViewingState(LiveViewingState state)
    {
        if(!SettingsManager.Loaded)
        {
            return;
        }

        if(state != null && state.Active)
        {
            KeepActiveFrameRate();
        }
        ApplyFrameLimiter(false);
    }


    private void Start()
    {
        lastInputMousePosition = Input.mousePosition;
        KeepActiveFrameRate();

        bool foundBloom = bloomVolume.profile.TryGet<Bloom>(out bloom);
        if(foundBloom)
        {
            defaultBloomStrength = bloom.intensity.value;
        }
        else
        {
            Debug.LogWarning("Unable to find bloom post processing effect!");
        }

        SettingsManager.OnSettingsUpdated += UpdateGraphicsSettings;
        TimeManager.OnPlayingChanged += UpdatePlaying;
        ReplayManager.OnLiveViewingStateUpdated += UpdateLiveViewingState;
        if(SettingsManager.Loaded)
        {
            UpdateGraphicsSettings("all");
        }
    }


    private void Update()
    {
        UpdateAdaptiveFrameRate();
    }


    private void OnDestroy()
    {
        SettingsManager.OnSettingsUpdated -= UpdateGraphicsSettings;
        TimeManager.OnPlayingChanged -= UpdatePlaying;
        ReplayManager.OnLiveViewingStateUpdated -= UpdateLiveViewingState;
    }
}
