<script>
  // A control-bar tooltip: the control's name, an optional line of detail
  // and its keyboard shortcut(s). `trigger` renders the control itself and
  // must spread the props it is handed (bits-ui's trigger wiring).
  import * as Tooltip from "$lib/components/ui/tooltip/index.js";
  import { Kbd } from "$lib/components/ui/kbd/index.js";

  let { label, desc = "", keys = [], trigger } = $props();
</script>

<Tooltip.Root>
  <Tooltip.Trigger>
    {#snippet child({ props })}{@render trigger(props)}{/snippet}
  </Tooltip.Trigger>
  <Tooltip.Content class="ui max-w-64" sideOffset={6}>
    <span class="flex flex-col gap-0.5">
      <span class="font-medium">{label}</span>
      {#if desc}<span class="leading-snug opacity-70">{desc}</span>{/if}
    </span>
    {#each keys as k (k)}<Kbd>{k}</Kbd>{/each}
  </Tooltip.Content>
</Tooltip.Root>
