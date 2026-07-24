using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Audio;

public class HitSoundManager : MonoBehaviour
{
#if !UNITY_WEBGL || UNITY_EDITOR
    public const float SoundOffset = 0.185f;

    public static AudioClip HitSound;
    public static AudioClip BadHitSound;
    public static List<ScheduledSound> scheduledSounds = new List<ScheduledSound>();
#endif

    public static bool RandomPitch
    {
        get
        {
            EnsureSettingsCached();
            return randomPitch;
        }
    }

    public static bool Spatial
    {
        get
        {
            EnsureSettingsCached();
            return spatial;
        }
    }

    public static float ScheduleBuffer
    {
        get
        {
            EnsureSettingsCached();
            return scheduleBuffer;
        }
    }

    public static bool DynamicPriority
    {
        get
        {
            EnsureSettingsCached();
            return dynamicPriority;
        }
    }

    public static bool HitSoundsEnabled
    {
        get
        {
            EnsureSettingsCached();
            return hitSoundsEnabled;
        }
    }

    public static bool ChainSoundsEnabled
    {
        get
        {
            EnsureSettingsCached();
            return chainSoundsEnabled;
        }
    }

    private static readonly string[] cachedSettings = new string[]
    {
        "randomhitsoundpitch",
        "spatialhitsounds",
        "hitsoundbuffer",
        "dynamicsoundpriority",
        "enablehitsound",
        "hitsoundvolume",
        "chainvolume",
        "mutemisses",
        "usebadhitsound"
    };

    private static bool randomPitch;
    private static bool spatial;
    private static float scheduleBuffer;
    private static bool dynamicPriority;
    private static bool hitSoundsEnabled;
    private static bool chainSoundsEnabled;
    private static bool muteMisses;
    private static bool useBadHitSound;
    private static bool settingsCached;

#if !UNITY_WEBGL || UNITY_EDITOR
    public float HitSoundVolume
    {
        set
        {
            value = (float)System.Math.Round(value, 2);
            if(value == 0)
            {
                value = 0.0001f;
            }
            //Logarithmic scaling makes volume slider feel more natural to the user
            hitSoundMixer.SetFloat("Volume", Mathf.Log10(value) * 20);
        }
    }

    public float ChainVolume
    {
        set
        {
            value = (float)System.Math.Round(value, 2);
            if(value.Approximately(0f))
            {
                value = 0.0001f;
            }
            //Logarithmic scaling makes volume slider feel more natural to the user
            hitSoundMixer.SetFloat("ChainVolume", Mathf.Log10(value) * 20);
        }
    }

    [SerializeField] private AudioMixer hitSoundMixer;
    [SerializeField] private AudioClip defaultHitsound;
    [SerializeField] private AudioClip defaultBadHitsound;
#endif


    private static void EnsureSettingsCached()
    {
        if(settingsCached || !SettingsManager.Loaded)
        {
            return;
        }

        UpdateSettings("all");
    }


    private static void UpdateSettings(string setting)
    {
        if(!SettingsManager.Loaded)
        {
            settingsCached = false;
            return;
        }

        if(setting != "all" && !cachedSettings.Contains(setting))
        {
            return;
        }

        randomPitch = SettingsManager.GetBool("randomhitsoundpitch");
        spatial = SettingsManager.GetBool("spatialhitsounds");
        scheduleBuffer = Mathf.Clamp(SettingsManager.GetFloat("hitsoundbuffer", false), 0f, 0.5f);
        dynamicPriority = SettingsManager.GetBool("dynamicsoundpriority", false);
        muteMisses = SettingsManager.GetBool("mutemisses");
        useBadHitSound = SettingsManager.GetBool("usebadhitsound");

        hitSoundsEnabled = SettingsManager.GetBool("enablehitsound") && SettingsManager.GetFloat("hitsoundvolume") > 0f;
        chainSoundsEnabled = hitSoundsEnabled && SettingsManager.GetFloat("chainvolume") > 0f;
        settingsCached = true;
    }


    public static void ScheduleHitsound(HitSoundEmitter emitter)
    {
        EnsureSettingsCached();

        if(!hitSoundsEnabled || (emitter.GetType() == typeof(ChainLink) && !chainSoundsEnabled))
        {
            return;
        }

        if(!emitter.WasHit && !emitter.WasBadCut && muteMisses)
        {
            //This note was missed and shouldn't play a sound
            return;
        }

#if !UNITY_WEBGL || UNITY_EDITOR
        AudioSource source = emitter.source;

        source.enabled = true;
        source.Stop();
        source.volume = 1f;

        if(randomPitch)
        {
            source.pitch = Random.Range(0.95f, 1.05f);
        }
        else source.pitch = 1f;

        if(emitter.WasBadCut && useBadHitSound)
        {
            source.clip = BadHitSound;
        }
        else source.clip = HitSound;
        
        if(spatial)
        {
            const float spread = 120f;

            source.spatialBlend = 1f;
            source.spread = spread;

            //Findall my behated
            List<ScheduledSound> existingSounds = scheduledSounds.FindAll(x => ObjectManager.CheckSameTime(x.time, emitter.Time));
            if(existingSounds.Count > 0)
            {
                //This sound has already been scheduled, every source should match pitch and have reduced volume
                const float volumeFalloff = 0.6f;
                foreach(ScheduledSound existingSound in existingSounds)
                {
                    existingSound.source.volume *= volumeFalloff;
                }
                source.volume *= volumeFalloff;

                source.pitch = existingSounds[0].source.pitch;
            }
        }
        else 
        {
            source.spatialBlend = 0f;
            source.spread = 0f;

            if(scheduledSounds.Any(x => ObjectManager.CheckSameTime(x.time, emitter.Time) && x.source.clip == source.clip))
            {
                //This sound has already been scheduled and we aren't using spatial audio, so sounds shouldn't stack
                source.enabled = false;
                return;
            }
        }

        ScheduledSound sound = new ScheduledSound
        {
            parentList = scheduledSounds,
            source = source,
            //If the object is a bomb, play the hitsound at the actual time it was hit
            time = emitter.GetType() == typeof(Bomb) ? emitter.Time - emitter.HitOffset : emitter.Time
        };
        scheduledSounds.Add(sound);
#else
        bool badCut = emitter.WasBadCut && useBadHitSound;
        bool chainLink = emitter.GetType() == typeof(ChainLink);
        float time = emitter.GetType() == typeof(Bomb) ? emitter.Time - emitter.HitOffset : emitter.Time;
        float pitch = randomPitch ? Random.Range(0.95f, 1.05f) : 1f;

        WebHitSoundController.CreateHitSound(badCut, chainLink, time, pitch);
#endif
    }


    public static void ClearScheduledSounds()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        for(int i = scheduledSounds.Count - 1; i >= 0; i--)
        {
            scheduledSounds[i].Destroy();
        }
#else
        WebHitSoundController.ClearScheduledSounds();
#endif
    }


    public void UpdatePlaying(bool playing)
    {
        if(!playing)
        {
            ClearScheduledSounds();
        }
    }


    public void UpdateTimeScale(float newScale)
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        float currentTime = SongManager.GetSongTime();
        for(int i = scheduledSounds.Count - 1; i >= 0; i--)
        {
            ScheduledSound sound = scheduledSounds[i];

            //If the sound is already scheduled, try to reschedule it with the correct timing
            if(sound.scheduled)
            {
                sound.source.Stop();
                sound.scheduled = false;

                sound.UpdateTime(currentTime);
            }
        }
#else
        WebHitSoundController.RescheduleHitsounds();
#endif
    }


#if !UNITY_WEBGL || UNITY_EDITOR
    private void Update()
    {
        float currentTime = SongManager.GetSongTime();
        for(int i = scheduledSounds.Count - 1; i >= 0; i--)
        {
            //Update each sound's priority and queue and such
            scheduledSounds[i].UpdateTime(currentTime);
        }
    }
#endif


    private void Start()
    {
        SettingsManager.OnSettingsUpdated -= UpdateSettings;
        SettingsManager.OnSettingsUpdated += UpdateSettings;
        UpdateSettings("all");

        TimeManager.OnPlayingChanged += UpdatePlaying;
        TimeSyncHandler.OnTimeScaleChanged += UpdateTimeScale;

#if UNITY_WEBGL && !UNITY_EDITOR
        WebHitSoundController.Init();
#endif
    }


#if !UNITY_WEBGL || UNITY_EDITOR
    private void Awake()
    {
        HitSound = defaultHitsound;
        BadHitSound = defaultBadHitsound;
    }
#endif
}


#if !UNITY_WEBGL || UNITY_EDITOR
public class ScheduledSound
{
    public List<ScheduledSound> parentList;
    public AudioSource source;
    public float time;
    public bool scheduled;


    public void Destroy()
    {
        if(source != null)
        {
            source.Stop();
        }

        parentList.Remove(this);
    }


    public void UpdateTime(float currentTime)
    {
        if(source == null || !source.isActiveAndEnabled)
        {
            Destroy();
            return;
        }

        //Account for time scale and sound offset
        float timeDifference = (time - currentTime) / TimeSyncHandler.TimeScale;

        if(!scheduled)
        {
            if(timeDifference <= 0)
            {
                //The sound should already be playing by this point
                //Trying to schedule now would just make it off-time and wouldn't be worth it
                Destroy();
                return;
            }

            float scheduleIn = timeDifference - (HitSoundManager.SoundOffset / source.pitch);
            if(!source.isPlaying && scheduleIn <= HitSoundManager.ScheduleBuffer)
            {
                if(scheduleIn <= 0)
                {
                    //The sound should already be playing the windup
                    //Instead, schedule the sound exactly on beat with no offset
                    source.time = HitSoundManager.SoundOffset;
                    source.PlayDelayed(timeDifference);
                }
                else
                {
                    source.time = 0;
                    source.PlayDelayed(scheduleIn);
                }

                scheduled = true;
            }
        }
        else 
        {
            if(HitSoundManager.DynamicPriority)
            {
                //Dynamically set sound priority so playing sounds don't get overridden by scheduled sounds
                //Thanks galx for making this code
                if(currentTime - time > 0)
                {
                    const float priorityFalloff = 384;
                    source.priority = Mathf.Clamp(1 + Mathf.RoundToInt((float)(currentTime - time) * priorityFalloff), 1, 254);
                }
                else
                {
                    const float priorityRampup = 192;
                    source.priority = Mathf.Clamp(1 + Mathf.RoundToInt((float)(time - currentTime) * priorityRampup), 1, 255);
                }
            }
            else source.priority = 100;
        }
    }
}
#endif
