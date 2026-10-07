<script>
  // The replay viewer's bottom bar (shadcn-svelte + Lucide; Tailwind scoped
  // to `.ui` in src/app.css). A reskin of the old button row: every control
  // calls exactly what its old button / range did, the keyboard shortcuts
  // stay in App.svelte's onKey, and the tip trail's switch + length persist
  // through setTipTrail / setTipTrailS as before.
  //
  //   row 1  the timeline: TorqueSeek, a seek bar drawn as the replay's
  //          per-tick torque profile, full width
  //   row 2  play/pause, restart | slower, speed, faster | [view toggles] | sound
  //
  // Narrow screens (< 640 px) keep transport, speed and mute in the row and
  // move the view toggles, their sliders and the volume slider into a
  // "more controls" popover.
  import { MediaQuery } from "svelte/reactivity";
  import Play from "@lucide/svelte/icons/play";
  import Pause from "@lucide/svelte/icons/pause";
  import RotateCcw from "@lucide/svelte/icons/rotate-ccw";
  import Minus from "@lucide/svelte/icons/minus";
  import Plus from "@lucide/svelte/icons/plus";
  import RectangleGoggles from "@lucide/svelte/icons/rectangle-goggles";
  import PersonStanding from "@lucide/svelte/icons/person-standing";
  import Gauge from "@lucide/svelte/icons/gauge";
  import Eye from "@lucide/svelte/icons/eye";
  import Sparkles from "@lucide/svelte/icons/sparkles";
  import Volume from "@lucide/svelte/icons/volume";
  import Volume2 from "@lucide/svelte/icons/volume-2";
  import VolumeX from "@lucide/svelte/icons/volume-x";
  import Ellipsis from "@lucide/svelte/icons/ellipsis";
  import { Button } from "$lib/components/ui/button/index.js";
  import { Toggle } from "$lib/components/ui/toggle/index.js";
  import { Slider } from "$lib/components/ui/slider/index.js";
  import { Separator } from "$lib/components/ui/separator/index.js";
  import { Kbd } from "$lib/components/ui/kbd/index.js";
  import * as Tooltip from "$lib/components/ui/tooltip/index.js";
  import * as Popover from "$lib/components/ui/popover/index.js";
  import BarTip from "./BarTip.svelte";
  import TorqueSeek from "./TorqueSeek.svelte";
  import { audio, pb, pose, restart, setPlaying, setSpeed,
    setTipTrail, setTipTrailS, togglePose } from "./state.svelte.js";
  import { TIP_TRAIL_MAX_S, TIP_TRAIL_MIN_S } from "./tiptrail.js";
  import { enterPlay } from "./playctl.js";
  import { FLUX, POSE } from "./build.js";

  const narrow = new MediaQuery("max-width: 639.98px", false);
  let moreOpen = $state(false);

  const speedLabel = $derived(
    pb.speed.toFixed(2).replace(/0+$/, "").replace(/\.$/, ".0") + "×");

  const ready = $derived(audio.status === "ready");
  const audioLabel = $derived(
    audio.status === "ready" ? (audio.song || "song")
    : audio.status === "loading" ? (audio.msg || "song…")
    : audio.status === "error" ? "no audio"
    : audio.status === "none" ? "no song"
    : "");

  // the view toggles, in the old bar's order; `slider` is the setting that
  // shows next to a toggle while it is on
  const toggles = $derived([
    POSE && { id: "pose", label: "pose", keys: ["P"], icon: PersonStanding,
      desc: "pose the body by hand: root and all 29 joints",
      on: pose.on, set: (v) => { if (v !== pose.on) togglePose(); } },
    { id: "gauges", label: "gauges", keys: ["G"], icon: Gauge,
      desc: "per-joint dials, torque bars and foot forces",
      on: pb.gauges, set: (v) => (pb.gauges = v) },
    { id: "first-person", label: "first-person", keys: ["F"], icon: Eye,
      desc: "camera at the player's head, looking straight down the track",
      on: pb.firstPerson, set: (v) => (pb.firstPerson = v), slider: fovSlider },
    { id: "tip-trail", label: "tip trail", keys: ["T"], icon: Sparkles,
      desc: "a crisp line tracing each blade tip's recent path",
      on: pb.tipTrail, set: (v) => setTipTrail(v), slider: trailSlider },
  ].filter(Boolean));

  // an "on" state that reads at a glance over the scene: accent tint + ring
  const TOGGLE_ON = "text-muted-foreground aria-pressed:bg-on/20 aria-pressed:text-on " +
    "aria-pressed:ring-1 aria-pressed:ring-inset aria-pressed:ring-on/60";
</script>

{#snippet fovSlider()}
  <div id="fp" class="flex items-center gap-2 text-xs tabular-nums text-muted-foreground">
    <span title="first-person vertical field of view">fov</span>
    <Slider type="single" min={40} max={110} step={1} class="w-20 py-1.5"
      thumbLabel="first-person field of view" valueText="{pb.fpFov}°"
      bind:value={() => pb.fpFov, (v) => (pb.fpFov = v)} />
    <span class="min-w-[4ch] text-foreground">{pb.fpFov}°</span>
  </div>
{/snippet}

{#snippet trailSlider()}
  <div id="tt" class="flex items-center gap-2 text-xs tabular-nums text-muted-foreground">
    <span title="how much of the tip's path the line shows, in replay seconds">len</span>
    <Slider type="single" min={TIP_TRAIL_MIN_S} max={TIP_TRAIL_MAX_S} step={0.05}
      class="w-20 py-1.5" thumbLabel="tip trail length"
      valueText="{pb.tipTrailS.toFixed(2)} s"
      bind:value={() => pb.tipTrailS, (v) => setTipTrailS(v)} />
    <span class="min-w-[4ch] text-foreground">{pb.tipTrailS.toFixed(2)} s</span>
  </div>
{/snippet}

{#snippet volumeSlider(cls)}
  <Slider type="single" min={0} max={1} step={0.02} class={cls}
    thumbLabel="volume" valueText="{Math.round(audio.vol * 100)}%"
    bind:value={() => audio.vol, (v) => (audio.vol = v)} />
{/snippet}

{#snippet soundToggle(props = {})}
  <Toggle {...props} size="sm" class="w-8 px-0 {TOGGLE_ON}" aria-label="sound"
    disabled={!ready}
    bind:pressed={() => ready && !audio.muted, () => (audio.muted = !audio.muted)}>
    {#if !ready}<Volume />{:else if audio.muted}<VolumeX />{:else}<Volume2 />{/if}
  </Toggle>
{/snippet}

<Tooltip.Provider delayDuration={250}>
  <div id="bar" class="ui pointer-events-auto flex w-full max-w-[52rem] flex-col gap-1
    rounded-[10px] border border-[rgba(58,64,80,0.6)] bg-[rgba(17,19,25,0.72)] px-3 py-1.5
    backdrop-blur-[6px]">
    <div id="seek" class="px-1">
      <TorqueSeek compact={narrow.current} />
    </div>

    <div class="flex flex-wrap items-center justify-between gap-x-2 gap-y-1">
      <div class="flex items-center gap-1">
        <BarTip label={pb.playing ? "pause" : "play"} keys={["Space"]}>
          {#snippet trigger(props)}
            <Button {...props} variant="ghost" size="icon-sm"
              aria-label={pb.playing ? "pause" : "play"}
              onclick={() => setPlaying(!pb.playing)}>
              {#if pb.playing}<Pause class="fill-current" />{:else}<Play class="fill-current" />{/if}
            </Button>
          {/snippet}
        </BarTip>
        <BarTip label="restart" keys={["R"]}>
          {#snippet trigger(props)}
            <Button {...props} variant="ghost" size="icon-sm" aria-label="restart"
              onclick={restart}><RotateCcw /></Button>
          {/snippet}
        </BarTip>
        <Separator orientation="vertical" class="mx-1 data-vertical:h-4 data-vertical:self-center" />
        <BarTip label="slower" desc="halve the playback speed" keys={["−"]}>
          {#snippet trigger(props)}
            <Button {...props} variant="ghost" size="icon-sm" aria-label="slower"
              onclick={() => setSpeed(pb.speed / 2)}><Minus /></Button>
          {/snippet}
        </BarTip>
        <span id="speed" class="min-w-[3.5ch] text-center text-xs tabular-nums"
          title="playback speed">{speedLabel}</span>
        <BarTip label="faster" desc="double the playback speed" keys={["+"]}>
          {#snippet trigger(props)}
            <Button {...props} variant="ghost" size="icon-sm" aria-label="faster"
              onclick={() => setSpeed(pb.speed * 2)}><Plus /></Button>
          {/snippet}
        </BarTip>
      </div>

      <div class="flex flex-wrap items-center justify-end gap-1">
        {#if !narrow.current}
          {#if FLUX}
            <BarTip label="play Flux"
              desc="play a Flux chart yourself in VR — the same judge the env uses">
              {#snippet trigger(props)}
                <Button {...props} variant="ghost" size="icon-sm" aria-label="play Flux"
                  onclick={enterPlay}><RectangleGoggles /></Button>
              {/snippet}
            </BarTip>
          {/if}
          {#each toggles as t (t.id)}
            <BarTip label={t.label} desc={t.desc} keys={t.keys}>
              {#snippet trigger(props)}
                <Toggle {...props} size="sm" class="w-8 px-0 {TOGGLE_ON}" aria-label={t.label}
                  data-toggle={t.id} bind:pressed={() => t.on, (v) => t.set(v)}>
                  <t.icon />
                </Toggle>
              {/snippet}
            </BarTip>
            {#if t.on && t.slider}{@render t.slider()}{/if}
          {/each}
          <Separator orientation="vertical" class="mx-1 data-vertical:h-4 data-vertical:self-center" />
        {/if}

        {#if pb.rep}
          <span id="snd" class="flex items-center gap-1.5"
            title={ready ? undefined : audio.msg || audio.song}>
            {#if ready}
              <BarTip label={audio.muted ? "unmute" : "mute"} desc={audio.song}>
                {#snippet trigger(props)}{@render soundToggle(props)}{/snippet}
              </BarTip>
            {:else}
              {@render soundToggle()}
            {/if}
            {#if !narrow.current}
              {#if ready}
                {@render volumeSlider("w-[70px] py-1.5")}
              {:else}
                <span class="max-w-[22ch] truncate text-xs text-muted-foreground">{audioLabel}</span>
              {/if}
            {/if}
          </span>
        {/if}

        {#if narrow.current}
          <Popover.Root bind:open={moreOpen}>
            <Popover.Trigger>
              {#snippet child({ props })}
                <Button {...props} variant="ghost" size="icon-sm" aria-label="more controls">
                  <Ellipsis />
                </Button>
              {/snippet}
            </Popover.Trigger>
            <!-- anchored to the whole bar, so it opens above it and never
                 covers the timeline -->
            <Popover.Content id="more" side="top" align="end" sideOffset={8}
              customAnchor="#bar" class="ui w-[min(18rem,calc(100vw-24px))] gap-2 p-2">
              {#if FLUX}
                <Button variant="ghost" size="sm" class="justify-start gap-2" aria-label="play Flux"
                  onclick={() => { moreOpen = false; enterPlay(); }}>
                  <RectangleGoggles /> play Flux
                </Button>
              {/if}
              {#each toggles as t (t.id)}
                <Toggle size="sm" class="justify-start gap-2 {TOGGLE_ON}" aria-label={t.label}
                  data-toggle={t.id} bind:pressed={() => t.on, (v) => t.set(v)}>
                  <t.icon /> {t.label}
                  <Kbd class="ml-auto [@media(pointer:coarse)]:hidden">{t.keys[0]}</Kbd>
                </Toggle>
                {#if t.on && t.slider}
                  <div class="pb-1 pl-8">{@render t.slider()}</div>
                {/if}
              {/each}
              {#if pb.rep}
                <Separator />
                <div class="flex items-center gap-2 px-2 text-xs text-muted-foreground">
                  <span>volume</span>
                  {#if ready}
                    {@render volumeSlider("flex-1 py-2")}
                  {:else}
                    <span class="truncate">{audioLabel}</span>
                  {/if}
                </div>
              {/if}
            </Popover.Content>
          </Popover.Root>
        {/if}
      </div>
    </div>
  </div>
</Tooltip.Provider>
