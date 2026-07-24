using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Assets.__Scripts.Loading.Replays.PP;
using Assets.__Scripts.Loading.Replays.ScoreSaber.Utils;
using UnityEngine;

public class ScoreSaberSource : ReplaySource
{
    private const string BaseURLEnv = "ARCVIEWER_SCORESABER_BASE_URL";
    private const string ApiURLEnv = "ARCVIEWER_SCORESABER_API_URL";
    private const string DefaultBaseURL = "https://scoresaber.com/";

    public static ScoreSaberPPHandler PPHandler { get; private set; }

    public override ReplaySourceType SourceType => ReplaySourceType.ScoreSaber;
    public override string Name => "ScoreSaber";
    public override string[] InputPrefixes => new[] { "ss:", "scoresaber:" };
    public override string BaseURL => GetURL(BaseURLEnv, DefaultBaseURL);
    public override string ApiURL => GetURL(ApiURLEnv, GetPathURL(BaseURL, "api/v2/"));
    public override string[] CorsURLs => new[] { BaseURL, ApiURL, "https://watch.scoresaber.com", "https://cdn.scoresaber.com" };

#if UNITY_WEBGL && !UNITY_EDITOR
    [DllImport("__Internal")]
    private static extern string GetArcViewerEnv(string name);
#endif


    public override bool MatchesHost(string host)
    {
        return host.Equals("scoresaber.com", StringComparison.InvariantCultureIgnoreCase)
            || host.Equals("watch.scoresaber.com", StringComparison.InvariantCultureIgnoreCase);
    }


    public override ReplaySourceInfo CreateInfo()
    {
        return CreateInfo(null);
    }


    private static string GetURL(string envName, string defaultURL)
    {
        string url = null;
#if UNITY_WEBGL && !UNITY_EDITOR
        url = GetArcViewerEnv(envName);
#else
        url = Environment.GetEnvironmentVariable(envName);
#endif

        if(string.IsNullOrWhiteSpace(url))
        {
            url = defaultURL;
        }

        return url.EndsWith("/") ? url : $"{url}/";
    }


    private static string GetPathURL(string baseURL, string path)
    {
        return new Uri(new Uri(baseURL), path).ToString();
    }


    public ReplaySourceInfo CreateInfo(ScoreSaberScoreResponse response)
    {
        ScoreSaberPlayerInfo player = response?.GetPlayer();
        ScoreSaberLeaderboardInfo leaderboard = response?.leaderboard;
        ReplaySourceInfo info = new ReplaySourceInfo
        {
            SourceType = ReplaySourceType.ScoreSaber,
            MapHash = leaderboard?.map?.hash,
            DifficultyRaw = leaderboard?.difficulty?.difficulty ?? 0,
            Characteristic = CharacteristicFromGameMode(leaderboard?.difficulty?.gameMode),
            MapID = leaderboard?.map?.bsid,
            PlayerName = player?.name,
            PlayerID = player?.id,
            AvatarURL = player?.avatar
        };

        if(!string.IsNullOrEmpty(info.PlayerID))
        {
            info.PlayerProfileURL = $"{BaseURL}u/{info.PlayerID}";
        }

        if(leaderboard?.map?.id > 0 && leaderboard.id > 0)
        {
            info.LeaderboardURL = $"{BaseURL}map/{leaderboard.map.id}/difficulty/{leaderboard.id}";
        }

        info.LoadSourceData = replay => LoadSourceDataAsync(info, replay);

        if(response?.leaderboard?.realm != null)
        {
            if(PPHandler == null)
            {
                PPHandler = new ScoreSaberPPHandler(response.leaderboard.realm.stars);
                PPManager.RegisterProvider(PPHandler);
            }
            else
            {
                PPHandler.SetScoreSaberStars(response.leaderboard.realm.stars);
            }
        }

        return info;
    }


    public override async Task<ResolvedScore> ResolveScoreAsync(string scoreID, string mapURL, string mapID, bool showErrors = true)
    {
        ScoreSaberScoreResponse apiResponse = await ScoreSaberApi.ScoreFromID(scoreID, showErrors);
        if(apiResponse == null || apiResponse.score == null || !apiResponse.score.hasReplay)
        {
            return null;
        }

        ReplaySourceInfo sourceInfo = CreateInfo(apiResponse);

        if(string.IsNullOrEmpty(mapID))
        {
            mapID = apiResponse.leaderboard?.map?.bsid;
        }

        return new ResolvedScore
        {
            ReplayURL = ScoreSaberApi.ReplayURLFromID(apiResponse.score?.id > 0 ? apiResponse.score.id.ToString() : scoreID),
            MapURL = mapURL,
            MapID = mapID,
            SourceInfo = sourceInfo
        };
    }


    public async Task LoadSourceDataAsync(ReplaySourceInfo source, Replay replay)
    {
        if(source == null || replay == null)
        {
            return;
        }

        if(!object.ReferenceEquals(ReplayManager.SourceInfo, source))
        {
            return;
        }

        if(!string.IsNullOrEmpty(replay.info?.playerID))
        {
            source.PlayerID = replay.info.playerID;
            source.PlayerProfileURL = $"{BaseURL}u/{source.PlayerID}";
        }

        if(string.IsNullOrEmpty(source.AvatarURL)) return;

        string avatarUrl = source.AvatarURL;
#if UNITY_WEBGL && !UNITY_EDITOR
        avatarUrl = WebLoader.GetCorsURL(avatarUrl);
#endif
        byte[] avatarData = await ReplayLoader.DownloadAvatarData(avatarUrl);
        if(avatarData != null && avatarData.Length > 0)
        {
            ReplayManager.SetAvatarImageData(avatarData);
        }
    }


    private static string CharacteristicFromGameMode(string gameMode)
    {
        if(string.IsNullOrEmpty(gameMode)) return null;
        if(!gameMode.StartsWith("Solo")) return gameMode;

        string characteristic = gameMode.Substring(4);
        return string.IsNullOrEmpty(characteristic) ? "Standard" : characteristic;
    }
}
