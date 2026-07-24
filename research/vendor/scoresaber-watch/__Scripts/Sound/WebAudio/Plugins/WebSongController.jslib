var SongController = {

    $ArcViewerSongAudioUnlock: {
        StartDelaySeconds: 0.035,

        InstallUnlockHandlers: function () {
            if (typeof window === 'undefined' || typeof document === 'undefined' || window.__arcViewerAudioUnlockInstalled) {
                return;
            }

            window.__arcViewerAudioUnlockInstalled = true;
            const unlock = () => {
                ArcViewerSongAudioUnlock.ResumeAudioContexts();
            };

            document.addEventListener("pointerdown", unlock, true);
            document.addEventListener("touchstart", unlock, true);
            document.addEventListener("keydown", unlock, true);
            document.addEventListener("click", unlock, true);
        },

        ResumeAudioContexts: function () {
            if (typeof SongCtx !== 'undefined' && SongCtx.state === "suspended") {
                SongCtx.resume().catch(() => {});
            }
            if (typeof AudioCtx !== 'undefined' && AudioCtx.state === "suspended") {
                AudioCtx.resume().catch(() => {});
            }
        },

        ContextRunning: function (ctx) {
            return typeof ctx === 'undefined' || ctx.state === "running";
        },

        GetControllerSongTime: function (controller) {
            if (!controller) {
                return 0;
            }

            if (!controller.playing || typeof SongCtx === 'undefined' || SongCtx.state !== "running") {
                return controller.soundStartTime;
            }

            const contextPassedTime = Math.max(0, SongCtx.currentTime - controller.lastPlayed);
            const performancePassedTime = Math.max(0, (performance.now() / 1000) - controller.performanceStartTime);
            const passedTime = Math.min(contextPassedTime, performancePassedTime);
            return controller.soundStartTime + (passedTime * controller.playbackSpeed);
        },

        StopSource: function (clip) {
            try {
                clip.stop();
            }
            catch (err) {
                // browsers can throw if the source is still scheduled in a suspended context
                if (!err || err.name !== "InvalidStateError") {
                    throw err;
                }
            }
        },

        DisconnectNode: function (node, destination) {
            try {
                node.disconnect(destination);
            }
            catch (err) {
                // cleanup can race with source replacement when loading another song
                if (!err || err.name !== "InvalidAccessError") {
                    throw err;
                }
            }
        }
    },

    InitSongController__deps: ["$ArcViewerSongAudioUnlock"],
    DisposeSongClip__deps: ["$ArcViewerSongAudioUnlock"],
    UploadSongData__deps: ["$ArcViewerSongAudioUnlock"],
    StartSong__deps: ["$ArcViewerSongAudioUnlock"],
    StopSong__deps: ["$ArcViewerSongAudioUnlock"],
    GetSongTime__deps: ["$ArcViewerSongAudioUnlock"],
    SetSongPlaybackSpeed__deps: ["$ArcViewerSongAudioUnlock"],
    IsSongAudioReady__deps: ["$ArcViewerSongAudioUnlock"],
    RequestSongAudioUnlock__deps: ["$ArcViewerSongAudioUnlock"],

    InitSongController: function (volume) {
        if (typeof SongCtx === 'undefined') {
            SongCtx = new AudioContext();
        }

        this.playing = false;
        this.volume = Math.max(0, volume);
        this.playbackSpeed = 1;
        this.lastPlayed = SongCtx.currentTime;
        this.performanceStartTime = performance.now() / 1000;
        this.soundStartTime = 0;
        this.soundOffset = 0;

        this.gainNode = SongCtx.createGain();
        this.gainNode.gain.setValueAtTime(this.volume, SongCtx.currentTime);
        this.gainNode.connect(SongCtx.destination);

        ArcViewerSongAudioUnlock.InstallUnlockHandlers();
        ArcViewerSongAudioUnlock.ResumeAudioContexts();
    },

    DisposeSongClip: function () {
        if(!this.clip) {
            return;
        }

        if (this.playing) {
            ArcViewerSongAudioUnlock.StopSource(this.clip);
            ArcViewerSongAudioUnlock.DisconnectNode(this.clip, this.gainNode);
        }

        delete (this.clip.buffer);
        delete (this.clip);

        this.playing = false;
    },

    UploadSongData: function (data, dataLength, isOgg, gameObjectName, methodName) {
        //Convert the C# byte[] to an arraybuffer for audio decoding
        const byteArray = HEAPU8.slice(data, data + dataLength);

        gameObjectName = UTF8ToString(gameObjectName);
        methodName = UTF8ToString(methodName);

        let decodeFunction = (data, callback, errorCallback) => SongCtx.decodeAudioData(data, callback, errorCallback);
        if (isOgg && isSafari) {
            console.log("Using custom OggDecode module for Safari.");
            decodeFunction = (data, callback, errorCallback) => SongCtx.decodeOggData(data, callback, errorCallback);
        }

        if (this.clip) {
            if (this.playing) {
                ArcViewerSongAudioUnlock.StopSource(this.clip);
                ArcViewerSongAudioUnlock.DisconnectNode(this.clip, this.gainNode);
            }

            delete (this.clip.buffer);
            delete (this.clip);

            this.playing = false;
        }

        decodeFunction(byteArray.buffer,
            (decodedData) => {
                const newClip = SongCtx.createBufferSource();
                newClip.buffer = decodedData;

                this.clip = newClip;

                //Callback to C# says that decoding succeeded
                SendMessage(gameObjectName, methodName, 1);
            },
            (err) => {
                console.error("Error decoding audio data: " + err.err);

                //Callback to C# says that decoding failed
                SendMessage(gameObjectName, methodName, 0);
            });
    },

    SetSongOffset: function (offset) {
        this.soundOffset = offset;
    },

    StartSong: function (time) {
        if (this.playing) {
            return;
        }

        ArcViewerSongAudioUnlock.ResumeAudioContexts();

        this.gainNode.gain.cancelScheduledValues(SongCtx.currentTime);
        this.gainNode.gain.setValueAtTime(this.volume > 0 ? 0.0001 : 0, SongCtx.currentTime);
        if (this.volume > 0.0001) {
            this.gainNode.gain.exponentialRampToValueAtTime(this.volume, SongCtx.currentTime + 0.075);
        }
        else {
            this.gainNode.gain.setValueAtTime(this.volume, SongCtx.currentTime);
        }

        //Create a new clip to play because after it plays it's forfeit
        const newClip = SongCtx.createBufferSource();

        newClip.buffer = this.clip.buffer;
        newClip.playbackRate.value = this.playbackSpeed;
        newClip.connect(this.gainNode);

        const startDelay = ArcViewerSongAudioUnlock.StartDelaySeconds;
        this.lastPlayed = SongCtx.currentTime + startDelay;
        this.performanceStartTime = (performance.now() / 1000) + startDelay;
        this.soundStartTime = time;

        let startTime = time + this.soundOffset;
        if (startTime >= 0) {
            //Start the clip normally
            newClip.start(this.lastPlayed, startTime);
        }
        else {
            //Schedule the sound to be played ahead of time if playing at negative time
            if (this.playbackSpeed > 0) {
                //Account for playback speed, but don't divide by 0
                startTime /= this.playbackSpeed;
            }
            else startTime = 0;

            //Subtract startTime here because it's negative
            newClip.start(this.lastPlayed - startTime, 0);
        }

        delete (this.clip);
        this.clip = newClip;
        this.playing = true;
    },

    StopSong: function () {
        if (!this.playing) {
            return;
        }

        const stoppedTime = ArcViewerSongAudioUnlock.GetControllerSongTime(this);

        this.gainNode.gain.cancelScheduledValues(SongCtx.currentTime);
        this.gainNode.gain.setValueAtTime(this.volume, SongCtx.currentTime);
        if (this.volume > 0.0001) {
            this.gainNode.gain.exponentialRampToValueAtTime(0.0001, SongCtx.currentTime + 0.075);
        }

        const clip = this.clip;
        const wasPlaying = this.playing;
        const oldGain = this.gainNode;
        const oldCtx = SongCtx;

        //Create a new audio context to avoid desync stemming from AudioContext.currentTime
        SongCtx = new AudioContext();
        this.gainNode = SongCtx.createGain();
        this.gainNode.gain.setValueAtTime(this.volume > 0 ? 0.0001 : 0, SongCtx.currentTime);
        this.gainNode.connect(SongCtx.destination);

        this.soundStartTime = stoppedTime;
        this.lastPlayed = SongCtx.currentTime;
        this.performanceStartTime = performance.now() / 1000;
        this.playing = false;

        setTimeout(function () {
            if (clip && wasPlaying) {
                ArcViewerSongAudioUnlock.StopSource(clip);
                ArcViewerSongAudioUnlock.DisconnectNode(clip, oldGain);
            }

            ArcViewerSongAudioUnlock.DisconnectNode(oldGain, oldCtx.destination);

            delete (oldGain);
            oldCtx.close();
            delete (oldCtx);
        }, 75);
    },

    GetSongTime: function () {
        return ArcViewerSongAudioUnlock.GetControllerSongTime(this);
    },

    GetSongLength: function () {
        if (!this.clip) {
            return 0;
        }

        const buffer = this.clip.buffer;
        if (!buffer) {
            return 0;
        }

        return buffer.duration - this.soundOffset;
    },

    SetSongVolume: function (volume) {
        this.volume = Math.max(0, volume);

        if (this.playing) {
            this.gainNode.gain.cancelScheduledValues(SongCtx.currentTime);
            this.gainNode.gain.setValueAtTime(this.volume, SongCtx.currentTime);
        }
    },

    SetSongPlaybackSpeed: function (speed) {
        if (this.playing) {
            const startDelay = ArcViewerSongAudioUnlock.StartDelaySeconds;
            let time = ArcViewerSongAudioUnlock.GetControllerSongTime(this);

            let startTime = time + this.soundOffset;

            if (startTime < 0) {
                //The sound is scheduled, but hasn't played yet. Reschedule, accounting for the new playback speed
                //This fixes a very niche bug where changing playback speed with negative offset,
                //before the sound actually starts playing, causes it to desync
                const clip = this.clip;
                ArcViewerSongAudioUnlock.StopSource(clip);
                ArcViewerSongAudioUnlock.DisconnectNode(clip, this.gainNode);

                ArcViewerSongAudioUnlock.DisconnectNode(this.gainNode, SongCtx.destination);
                delete (this.gainNode);
                SongCtx.close();
                delete (SongCtx);

                SongCtx = new AudioContext();
                this.gainNode = SongCtx.createGain();
                this.gainNode.gain.setValueAtTime(this.volume, SongCtx.currentTime);
                this.gainNode.connect(SongCtx.destination);

                this.lastPlayed = SongCtx.currentTime + startDelay;
                this.performanceStartTime = (performance.now() / 1000) + startDelay;

                const newClip = SongCtx.createBufferSource();
                newClip.buffer = clip.buffer;
                newClip.playbackRate.value = speed;

                startTime /= speed;

                //Subtract startTime here because it's negative
                newClip.start(this.lastPlayed - startTime, 0);
                newClip.connect(this.gainNode);

                delete (this.clip);
                this.clip = newClip;
            }
            else {
                this.lastPlayed = SongCtx.currentTime + startDelay;
                this.performanceStartTime = (performance.now() / 1000) + startDelay;
                this.clip.playbackRate.setValueAtTime(speed, this.lastPlayed);
            }

            this.soundStartTime = time;
        }

        this.playbackSpeed = speed;
    },

    GetSongPlaybackSpeed: function () {
        return this.playbackSpeed;
    },

    IsSongAudioReady: function () {
        const songReady = ArcViewerSongAudioUnlock.ContextRunning(typeof SongCtx === 'undefined' ? undefined : SongCtx);
        const hitSoundReady = ArcViewerSongAudioUnlock.ContextRunning(typeof AudioCtx === 'undefined' ? undefined : AudioCtx);
        return songReady && hitSoundReady ? 1 : 0;
    },

    RequestSongAudioUnlock: function () {
        ArcViewerSongAudioUnlock.InstallUnlockHandlers();
        ArcViewerSongAudioUnlock.ResumeAudioContexts();
    }
};

mergeInto(LibraryManager.library, SongController);
