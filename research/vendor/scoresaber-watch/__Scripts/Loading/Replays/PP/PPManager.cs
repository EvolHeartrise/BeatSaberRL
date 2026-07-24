using System.Collections.Generic;

namespace Assets.__Scripts.Loading.Replays.PP
{
    public interface IPPProvider
    {
        bool CanHandle(ReplaySourceInfo source);
        float CalculatePP(float normalisedAccuracy, ReplaySourceInfo source);
        string GetShorthand();
    }

    public static class PPManager
    {
        private static readonly List<IPPProvider> providers = new List<IPPProvider>();

        public static void RegisterProvider(IPPProvider provider)
        {
            if(provider != null && !providers.Contains(provider))
            {
                providers.Add(provider);
            }
        }

        public static void UnregisterProvider(IPPProvider provider)
        {
            if(provider != null && providers.Contains(provider))
            {
                providers.Remove(provider);
            }
        }

        private static IPPProvider GetPreferredProvider()
        {
            int index = SettingsManager.GetInt("preferredppsource");

            if(index < 0 || index >= providers.Count)
                return null;

            return providers[index];
        }

        public static bool CanCalculatePP()
        {
            ReplaySourceInfo source = ReplayManager.SourceInfo;
            if(source == null) return false;
            foreach(var provider in providers)
            {
                if(provider == GetPreferredProvider() && provider.CanHandle(source))
                {
                    return true;
                }
            }
            return false;
        }

        public static string GetPPShorthand()
        {
            ReplaySourceInfo source = ReplayManager.SourceInfo;
            if(source == null) return "N/A";
            foreach(var provider in providers)
            {
                if(provider == GetPreferredProvider() && provider.CanHandle(source))
                {
                    return provider.GetShorthand();
                }
            }
            return "N/A";
        }

        public static float CalculatePP(float normalisedAccuracy, out string shorthand)
        {
            ReplaySourceInfo source = ReplayManager.SourceInfo;
            if(source == null)
            {
                shorthand = "N/A";
                return 0f;
            }

            foreach(var provider in providers)
            {
                if(provider == GetPreferredProvider() && provider.CanHandle(source))
                {
                    shorthand = provider.GetShorthand();
                    return provider.CalculatePP(normalisedAccuracy, source);
                }
            }
            shorthand = "N/A";
            return 0f;
        }
    }
}
