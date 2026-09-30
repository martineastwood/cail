// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Keep the homepage hero links and the deployed base path in one place.
// GitHub Pages serves project sites under /cail.
const base = '/cail';

export default defineConfig({
	site: 'https://martineastwood.github.io',
	base,
	integrations: [
		starlight({
			title: 'CAIL',
			description:
				'A typed C++ SDK for LLM providers, with a provider-neutral model and generation API.',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/martineastwood/cail' }],
			sidebar: [
				{ label: 'Overview', slug: 'index' },
				{ label: 'Install', slug: 'guides/install' },
				{ label: 'Quickstart', slug: 'guides/quickstart' },
				{ label: 'Structured outputs', slug: 'guides/structured-output' },
				{ label: 'Tools', slug: 'guides/tools' },
				{ label: 'Agent', slug: 'guides/agent' },
				{ label: 'Memory', slug: 'guides/memory' },
				{ label: 'Files, images, and PDFs', slug: 'guides/loaders' },
				{ label: 'Streaming', slug: 'guides/streaming' },
				{ label: 'Async and coroutines', slug: 'guides/async' },
				{ label: 'Request controls and results', slug: 'guides/request-controls' },
				{ label: 'Embeddings', slug: 'guides/embeddings' },
				{ label: 'Providers', slug: 'guides/providers' },
				{ label: 'Advanced usage', slug: 'guides/advanced' },
				{
					label: 'Contributing',
					items: [
						{ label: 'Build and test', slug: 'reference/build-and-test' },
						{ label: 'Compile time', slug: 'reference/compile-time' },
					],
				},
			],
		}),
	],
});
