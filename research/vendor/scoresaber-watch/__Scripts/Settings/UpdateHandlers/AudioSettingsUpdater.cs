using System.Linq;
using UnityEngine;

public class AudioSettingsUpdater : MonoBehaviour
{
    [SerializeField] private HitSoundManager hitSoundManager;

    [Space]
    [SerializeField] private AudioClip[] hitSounds;
    [SerializeField] private AudioClip[] badHitSounds;

    private string[] hitSoundSettings = new string[]
    {
        "hitsound",
        "spatialhitsounds",
        "randomhitsoundpitch",
        "usebadhitsound",
        "badhitsound",
        "mutemisses"
    };


    public void UpdateAudioSettings(string setting)
    {
        bool allSettings = setting == "all";

        if(allSettings || setting == "enablemusic")
        {
            bool musicEnabled = SettingsManager.GetBool("enablemusic");
            if(musicEnabled && SettingsManager.GetFloat("musicvolume") <= 0.001f)
            {
                // When unmuting with volume set to 0, just set volume to the default value
                float defaultMusicVolume;
                if(!Settings.GetDefaultSettings().Floats.TryGetValue("musicvolume", out defaultMusicVolume))
                {
                    defaultMusicVolume = 0.5f;
                }
                SettingsManager.SetRule("musicvolume", defaultMusicVolume);

                // Return here because we've just effectively re-called this method by setting volume
                return;
            }
        }

        if(allSettings || setting == "enablehitsound")
        {
            bool hitSoundsEnabled = SettingsManager.GetBool("enablehitsound");
            if(hitSoundsEnabled && SettingsManager.GetFloat("hitsoundvolume") <= 0.001f)
            {
                // When unmuting with volume set to 0, just set volume to the default value
                float defaultHitSoundVolume;
                if(!Settings.GetDefaultSettings().Floats.TryGetValue("hitsoundvolume", out defaultHitSoundVolume))
                {
                    defaultHitSoundVolume = 0.5f;
                }
                SettingsManager.SetRule("hitsoundvolume", defaultHitSoundVolume);

                // Return here because we've just effectively re-called this method by setting volume
                return;
            }
        }

        if(allSettings || setting == "musicvolume" || setting == "enablemusic")
        {
            SongManager.Instance.MusicVolume = SettingsManager.GetBool("enablemusic") ? Mathf.Clamp01(SettingsManager.GetFloat("musicvolume")) : 0f;
        }

        if(allSettings || setting == "hitsoundvolume" || setting == "chainvolume" || setting == "enablehitsound")
        {
            float hitSoundVolume = SettingsManager.GetBool("enablehitsound") ? Mathf.Clamp01(SettingsManager.GetFloat("hitsoundvolume")) : 0f;
#if !UNITY_WEBGL || UNITY_EDITOR
            hitSoundManager.HitSoundVolume = hitSoundVolume;
            hitSoundManager.ChainVolume = Mathf.Clamp01(SettingsManager.GetFloat("chainvolume")) * hitSoundVolume;
#else
            float chainSoundVolume = SettingsManager.GetFloat("chainvolume");

            WebHitSoundController.SetHitSoundVolume(hitSoundVolume);
            WebHitSoundController.SetChainSoundVolume(chainSoundVolume);

            bool hitSoundsOff = WebHitSoundController.CurrentHitSoundVolume < Mathf.Epsilon;
            bool chainSoundsOff = WebHitSoundController.CurrentChainSoundVolume < Mathf.Epsilon;
            if((hitSoundsOff && hitSoundVolume > Mathf.Epsilon) || (chainSoundsOff && chainSoundVolume > Mathf.Epsilon))
            {
                //Reschedule hitsounds if volume is going from 0 to greater than zero
                //This is necessary because web audio refuses to process audio with 0 volume
                HitSoundManager.ClearScheduledSounds();
                ObjectManager.Instance.RescheduleHitsounds(TimeManager.Playing);
            }

            WebHitSoundController.CurrentHitSoundVolume = hitSoundVolume;
            WebHitSoundController.CurrentChainSoundVolume = chainSoundVolume;
#endif
        }

        if(allSettings || hitSoundSettings.Contains(setting))
        {
            int hitsound = SettingsManager.GetInt("hitsound");
            hitsound = Mathf.Clamp(hitsound, 0, hitSounds.Length - 1);

            int badHitsound = SettingsManager.GetInt("badhitsound");
            badHitsound = Mathf.Clamp(badHitsound, 0, badHitSounds.Length - 1);

#if !UNITY_WEBGL || UNITY_EDITOR
            HitSoundManager.HitSound = hitSounds[hitsound];
            HitSoundManager.BadHitSound = badHitSounds[badHitsound];
#else
            WebHitSoundController.SetHitSound(hitsound);
            WebHitSoundController.SetBadHitSound(badHitsound);
#endif

            HitSoundManager.ClearScheduledSounds();
            ObjectManager.Instance.RescheduleHitsounds(TimeManager.Playing);
        }
    }
    

    private void Start()
    {
        SettingsManager.OnSettingsUpdated += UpdateAudioSettings;
        UpdateAudioSettings("all");
    }
}