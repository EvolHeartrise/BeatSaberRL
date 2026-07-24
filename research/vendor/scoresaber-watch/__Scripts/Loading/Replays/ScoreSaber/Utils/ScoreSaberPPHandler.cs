using System.Collections.Generic;
using System.Linq;
using Assets.__Scripts.Loading.Replays.PP;
using UnityEngine;

namespace Assets.__Scripts.Loading.Replays.ScoreSaber.Utils
{
    public class ScoreSaberPPHandler : IPPProvider
    {
        private float CurrentScoreSaberStars = 0;

        public bool IsScoreSaberRanked => CurrentScoreSaberStars > 0;

        private const float ScoreSaberStarValue = 42.117208413f;
        private readonly Dictionary<float, float> ScoreSaberV3 = new Dictionary<float, float>()
        {
            {1, 5.367394282890631f},
            {0.9995f, 5.019543595874787f},
            {0.999f, 4.715470646416203f},
            {0.99825f, 4.325027383589547f},
            {0.9975f, 3.996793606763322f},
            {0.99625f, 3.5526145337555373f},
            {0.995f, 3.2022017597337955f},
            {0.99375f, 2.9190155639254955f},
            {0.9925f, 2.685667856592722f},
            {0.99125f, 2.4902905794106913f},
            {0.99f, 2.324506282149922f},
            {0.9875f, 2.058947159052738f},
            {0.985f, 1.8563887693647105f},
            {0.9825f, 1.697536248647543f},
            {0.98f, 1.5702410055532239f},
            {0.9775f, 1.4664726399289512f},
            {0.975f, 1.3807102743105126f},
            {0.9725f, 1.3090333065057616f},
            {0.97f, 1.2485807759957321f},
            {0.965f, 1.1552120359501035f},
            {0.96f, 1.0871883573850478f},
            {0.955f, 1.0388633331418984f},
            {0.95f, 1f},
            {0.94f, 0.9417362980580238f},
            {0.93f, 0.9039994071865736f},
            {0.92f, 0.8728710341448851f},
            {0.91f, 0.8488375988124467f},
            {0.9f, 0.825756123560842f},
            {0.875f, 0.7816934560296046f},
            {0.85f, 0.7462290664143185f},
            {0.825f, 0.7150465663454271f},
            {0.8f, 0.6872268862950283f},
            {0.75f, 0.6451808210101443f},
            {0.7f, 0.6125565959114954f},
            {0.65f, 0.5866010012767576f},
            {0.6f, 0.18223233667439062f},
            {0, 0}
        };

        public ScoreSaberPPHandler(float ScoreSaberStars)
        {
            CurrentScoreSaberStars = ScoreSaberStars;
        }

        public void SetScoreSaberStars(float stars)
        {
            CurrentScoreSaberStars = stars;
        }

        public bool CanHandle(ReplaySourceInfo source)
        {
            return source != null && source.SourceType == ReplaySourceType.ScoreSaber && IsScoreSaberRanked;
        }

        float Lerp(float v0, float v1, float t)
        {
            return v0 + t * (v1 - v0);
        }

        float CalculatePPModifier(float lowerAcc, float lowerVal, float upperAcc, float upperVal, float acc)
        {
            if (Mathf.Approximately(upperAcc, lowerAcc)) return lowerVal;
            float t = (acc - lowerAcc) / (upperAcc - lowerAcc);
            t = Mathf.Clamp01(t);
            return Lerp(lowerVal, upperVal, t);
        }

        public string GetShorthand()
        {
            return "SS";
        }

        public float CalculatePP(float normalisedAccuracy, ReplaySourceInfo source)
        {
            if(!IsScoreSaberRanked)
            {
                return 0;
            }

            float ppValue = CurrentScoreSaberStars * ScoreSaberStarValue;

            var keys = ScoreSaberV3.Keys.OrderBy(k => k).ToArray();
            if(keys.Length == 0) return 0;

            if(normalisedAccuracy <= keys[0])
            {
                return ppValue * ScoreSaberV3[keys[0]];
            }
            if(normalisedAccuracy >= keys[keys.Length - 1])
            {
                return ppValue * ScoreSaberV3[keys[keys.Length - 1]];
            }

            for(int i = 0; i < keys.Length - 1; i++)
            {
                float lower = keys[i];
                float upper = keys[i + 1];
                if(normalisedAccuracy >= lower && normalisedAccuracy <= upper)
                {
                    float lowerVal = ScoreSaberV3[lower];
                    float upperVal = ScoreSaberV3[upper];
                    float multiplier = CalculatePPModifier(lower, lowerVal, upper, upperVal, normalisedAccuracy);
                    return ppValue * multiplier;
                }
            }

            return 0;
        }
    }
}
