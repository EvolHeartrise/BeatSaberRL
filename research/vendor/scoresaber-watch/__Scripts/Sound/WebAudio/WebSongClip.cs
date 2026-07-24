using System;

public class WebSongClip : IDisposable
{
    public bool IsPlaying => isPlaying;

    public float Length => length;

    private bool isPlaying = false;
    private float length = 0f;

    public float Time => WebSongController.GetSongTime();

    public WebSongClip()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        throw new InvalidOperationException("WebSongClip should only be used in WEBGL!");
#else

        // Make sure the audio controller was created
        WebSongController.Init();
#endif
    }


    public void Dispose()
    {
        length = 0f;
#if UNITY_WEBGL && !UNITY_EDITOR
        Stop();
#endif
        WebSongController.StopSong();
        WebSongController.DisposeSongClip();
    }


#if UNITY_WEBGL && !UNITY_EDITOR
    public void SetData(byte[] data, bool isOgg, Action<int> callback)
    {
        length = 0f;
        WebSongController.SetDataClip(data, isOgg, response =>
        {
            length = response > 0 ? WebSongController.GetSongLength() : 0f;
            callback?.Invoke(response);
        });
    }


    public void SetOffset(float offset)
    {
        WebSongController.SetSongOffset(offset);
        length = WebSongController.GetSongLength();
    }


    public void SetSpeed(float speed)
    {
        WebSongController.SetSongPlaybackSpeed(speed);
    }


    public void Play(float time = 0f)
    {
        if(isPlaying) return;

        WebSongController.StartSong(time);
        isPlaying = true;
    }


    public void Stop()
    {
        if(!isPlaying) return;

        WebSongController.StopSong();
        isPlaying = false;
    }
#endif
}