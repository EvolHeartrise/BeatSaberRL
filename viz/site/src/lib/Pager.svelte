<script>
	import * as Pagination from '$lib/components/ui/pagination/index.js';

	/** 0-based page over `total` rows, `per` per page. @type {{page: number, total: number, per: number}} */
	let { page = $bindable(), total, per } = $props();

	/** @param {number} p 1-based */
	function go(p) {
		page = p - 1;
		window.scrollTo({ top: 0 });
	}
</script>

<div class="flex flex-col items-center justify-between gap-3 pt-4 sm:flex-row">
	<p class="text-muted-foreground shrink-0 text-sm">
		{total === 0 ? 0 : (page * per + 1).toLocaleString('en-US')}–{Math.min(total, (page + 1) * per).toLocaleString('en-US')}
		of {total.toLocaleString('en-US')}
	</p>
	{#if total > per}
		<Pagination.Root count={total} perPage={per} siblingCount={1} bind:page={() => page + 1, go} class="mx-0 w-auto">
			{#snippet children({ pages, currentPage })}
				<Pagination.Content>
					<Pagination.Item><Pagination.Previous /></Pagination.Item>
					{#each pages as p (p.key)}
						{#if p.type === 'ellipsis'}
							<Pagination.Item class="hidden sm:block"><Pagination.Ellipsis /></Pagination.Item>
						{:else}
							<Pagination.Item class={currentPage === p.value ? '' : 'hidden sm:block'}>
								<Pagination.Link page={p} isActive={currentPage === p.value}>{p.value}</Pagination.Link>
							</Pagination.Item>
						{/if}
					{/each}
					<Pagination.Item><Pagination.Next /></Pagination.Item>
				</Pagination.Content>
			{/snippet}
		</Pagination.Root>
	{/if}
</div>
