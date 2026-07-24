using System;
using System.Threading.Tasks;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

#pragma warning disable CS4014
#pragma warning disable 1998
public class ReplayLoader
{
    public static async Task<Replay> ReplayFromDirectory(string directory)
    {
#if UNITY_WEBGL && !UNITY_EDITOR
        throw new InvalidOperationException("Loading from directory doesn't work in WebGL!");
#else
        try
        {
            byte[] replayData = await File.ReadAllBytesAsync(directory);
            return await DecodeReplayBytesAsync(replayData);
        }
        catch(Exception err)
        {
            Debug.LogWarning($"Failed to load replay with error: {err.Message}, {err.StackTrace}");
            ErrorHandler.Instance.QueuePopup(ErrorType.Error, "Failed to load replay file!");
            return null;
        }
#endif
    }


    public static Replay DecodeReplayBytes(byte[] data)
    {
        if(BsorDecoder.IsBsorV1Replay(data))
        {
            Debug.Log("Detected BsorV1 replay format.");
            return BsorDecoder.Decode(data);
        }

        // scoresaber replay (magic header check)
        if(ScoreSaberDecoder.IsScoreSaberReplay(data))
        {
            Debug.Log("Detected ScoreSaber replay format.");
            return ScoreSaberDecoder.DecodeKnownScoreSaberReplay(data);
        }

        // scoresaber legacy format
        if(ScoreSaberDecoder.IsLegacyScoreSaberReplay(data))
        {
            Debug.Log("Detected ScoreSaber legacy replay format.");
            Replay legacy = ScoreSaberLegacyDecoder.Decode(data);
            if(legacy != null) return legacy;
        }

        // fall back to BsorV1
        return BsorDecoder.Decode(data);
    }


    private static async Task<Replay> DecodeReplayBytesAsync(byte[] data)
    {
        if(BsorDecoder.IsBsorV1Replay(data))
        {
            Debug.Log("Detected BsorV1 replay format.");
            return BsorDecoder.Decode(data);
        }

        // scoresaber replay (magic header check)
        if(ScoreSaberDecoder.IsScoreSaberReplay(data))
        {
            Debug.Log("Detected ScoreSaber replay format.");
            return await ScoreSaberDecoder.DecodeKnownScoreSaberReplayAsync(data);
        }

        // scoresaber legacy format
        if(ScoreSaberDecoder.IsLegacyScoreSaberReplay(data))
        {
            Debug.Log("Detected ScoreSaber legacy replay format.");
            Replay legacy = ScoreSaberLegacyDecoder.Decode(data);
            if(legacy != null) return legacy;
        }

        // fall back to BsorV1
        return BsorDecoder.Decode(data);
    }


    public static async Task<Replay> ReplayFromStream(Stream replayStream)
    {
        byte[] data = await ReadReplayStreamBytes(replayStream);

        Replay replay = await DecodeReplayBytesAsync(data);
        if(replay != null)
        {
            if(replayStream.CanSeek)
            {
                replayStream.Seek(0, SeekOrigin.Begin);
            }
            return replay;
        }

        // fall back to BsorV1 async decoder
        if(replayStream.CanSeek)
        {
            replayStream.Seek(0, SeekOrigin.Begin);
        }

        AsyncBsorDecoder decoder = new AsyncBsorDecoder();
        (ReplayInfo, Task<Replay>) result = await decoder.StartDecodingStream(replayStream);

        if(result.Item2 == null)
        {
            return null;
        }

        Replay decodedReplay = await result.Item2;
        result.Item2.Dispose();

        if(replayStream.CanSeek)
        {
            replayStream.Seek(0, SeekOrigin.Begin);
        }

        return decodedReplay;
    }


    private static async Task<byte[]> ReadReplayStreamBytes(Stream replayStream)
    {
        if(replayStream is MemoryStream memoryStream && memoryStream.Position == 0 &&
            memoryStream.TryGetBuffer(out ArraySegment<byte> buffer) &&
            buffer.Offset == 0 && buffer.Count == buffer.Array.Length)
        {
            memoryStream.Seek(0, SeekOrigin.End);
            return buffer.Array;
        }

        if(replayStream.CanSeek)
        {
            long remaining = Math.Max(0, replayStream.Length - replayStream.Position);
            if(remaining <= int.MaxValue)
            {
                byte[] data = new byte[(int)remaining];
                int bytesRead = 0;
                while(bytesRead < data.Length)
                {
                    int read = await replayStream.ReadAsync(data, bytesRead, data.Length - bytesRead);
                    if(read == 0) break;
                    bytesRead += read;
                }

                if(bytesRead == data.Length)
                {
                    return data;
                }

                byte[] resizedData = new byte[bytesRead];
                Array.Copy(data, resizedData, bytesRead);
                return resizedData;
            }
        }

        using MemoryStream ms = new MemoryStream();
        await replayStream.CopyToAsync(ms);
        return ms.ToArray();
    }


    public static async Task<byte[]> DownloadAvatarData(string url)
    {
        if(string.IsNullOrEmpty(url))
        {
            Debug.LogWarning("Avatar url is empty!");
            return null;
        }
        Debug.Log($"Downloading avatar from {url}");

        try
        {
            using UnityWebRequest uwr = UnityWebRequest.Get(url);
            uwr.SendWebRequest();

            while(!uwr.isDone) await Task.Yield();

            if(uwr.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning($"Failed to get avatar image response with error: {uwr.error}");
                return null;
            }

            return uwr.downloadHandler.data;
        }
        catch(Exception err)
        {
            Debug.LogWarning($"Failed to get avatar image response with error: {err.Message}, {err.StackTrace}");
            return null;
        }
    }
}