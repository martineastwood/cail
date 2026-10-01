---
title: CAIL
description: A typed C++ SDK for LLM providers, with a provider-neutral model and generation API.
template: splash
hero:
  title: CAIL
  tagline: Typed C++23 calls to any LLM provider. One model, one generation API.
  actions:
    - text: Install
      link: /cail/guides/install/
      variant: primary
      icon: right-arrow
    - text: Quickstart
      link: /cail/guides/quickstart/
      variant: secondary
      icon: right-arrow
    - text: View on GitHub
      link: https://github.com/martineastwood/cail
      variant: secondary
      icon: external
---

<div class="landing-shell not-content">
  <p class="landing-lede">CAIL is a C++23 SDK with a static library. Write a generation request once and run it against OpenAI, Anthropic, Gemini, OpenRouter, Mistral, Azure Foundry, Charm Hyper, Ollama Cloud, OpenCode, or your own local server. Streaming, structured outputs, tools, agents with memory, embeddings, and local file or PDF attachments use the same provider-neutral API.</p>

  <section class="landing-terminal" aria-labelledby="landing-terminal-title">
    <div class="landing-terminal-bar">
      <div class="landing-terminal-dots" aria-hidden="true"><span></span><span></span><span></span></div>
      <span id="landing-terminal-title">your-app.cpp</span>
    </div>
    <pre class="not-content"><code><span class="kw">auto</span> response = cail::generate_text({
    .model = cail::openai(<span class="str">"gpt-6-luna"</span>),
    .prompt = <span class="str">"Summarize the main idea of this paragraph."</span>,
});
std::cout &lt;&lt; response-&gt;text;</code></pre>
  </section>

  <section class="landing-section" aria-labelledby="landing-api-title">
    <p class="landing-kicker">One API</p>
    <h2 id="landing-api-title">Provider-neutral by design</h2>
    <p class="landing-section-intro">You write against a single generation API. The provider is a value you pass in, so switching models or vendors is a one-line change.</p>
    <div class="landing-grid">
      <article class="landing-card">
        <span class="landing-card-index">01</span>
        <h3>Typed requests and results</h3>
        <p>Designated initializers for requests, <code>Result</code> values for responses. No JSON hand-rolling and no vendor structs in your code.</p>
      </article>
      <article class="landing-card">
        <span class="landing-card-index">02</span>
        <h3>Structured outputs</h3>
        <p>Describe a result with <code>cail::Field&lt;T&gt;</code> members. CAIL generates the JSON Schema and decodes the reply into your struct.</p>
      </article>
      <article class="landing-card">
        <span class="landing-card-index">03</span>
        <h3>Function tools and agents</h3>
        <p><code>cail::tool&lt;In, Out&gt;()</code> gives you typed argument decoding and a typed handler. Bundle tools with standing instructions in a <code>cail::Agent</code> and the tool loop, dispatch, and history are handled for you.</p>
      </article>
      <article class="landing-card">
        <span class="landing-card-index">04</span>
        <h3>Streaming everywhere</h3>
        <p>Text, reasoning, tool-call deltas, and usage arrive as typed <code>StreamEvent</code>s, with cancellation through <code>std::stop_token</code>.</p>
      </article>
      <article class="landing-card">
        <span class="landing-card-index">05</span>
        <h3>Files, memory, and embeddings</h3>
        <p>Load images and PDFs from disk, give agents durable conversation memory, embed text for search, and use <code>EmbeddingStore</code> for small in-memory collections.</p>
      </article>
      <article class="landing-card">
        <span class="landing-card-index">06</span>
        <h3>Async and coroutines</h3>
        <p><code>co_await</code> generation, streaming, and embeddings, or use completion callbacks when you integrate with an existing event loop.</p>
      </article>
    </div>
  </section>

  <section class="landing-section" aria-labelledby="landing-providers-title">
    <p class="landing-kicker">Providers</p>
    <h2 id="landing-providers-title">Hosted or on your own machine</h2>
    <p class="landing-section-intro">Every provider reads its API key from the environment by default, and each adapter reports what it can encode, decode, and stream.</p>
    <div class="landing-grid">
      <article class="landing-card">
        <span class="landing-card-index">01</span>
        <h3>Hosted providers</h3>
        <p>OpenAI, Anthropic, Gemini, OpenRouter, Mistral, Azure Foundry, Charm Hyper, Ollama Cloud, and OpenCode services.</p>
      </article>
      <article class="landing-card">
        <span class="landing-card-index">02</span>
        <h3>Local servers</h3>
        <p>Point <code>cail::local</code> at llama.cpp, Ollama, LM Studio, or vLLM. The same generation, streaming, tools, and structured output APIs.</p>
      </article>
      <article class="landing-card">
        <span class="landing-card-index">03</span>
        <h3>OpenAI-compatible</h3>
        <p>Any Chat Completions endpoint works through <code>cail::create_chat_completions</code>, with an optional session header.</p>
      </article>
      <article class="landing-card">
        <span class="landing-card-index">04</span>
        <h3>Build once, reuse</h3>
        <p>Install CAIL with CMake and link <code>cail::cail</code> in your application. Include the provider headers you need.</p>
      </article>
    </div>
  </section>
</div>

<style>
	.landing-shell {
		max-width: 72rem;
		margin: 0 auto;
	}
	.landing-lede {
		font-size: 1.125rem;
		max-width: 46rem;
		margin: 0 auto 3rem;
		text-align: center;
	}
	.landing-terminal {
		max-width: 42rem;
		margin: 0 auto 4rem;
		border: 1px solid var(--sl-color-gray-5);
		border-radius: 0.75rem;
		overflow: hidden;
		background: var(--sl-color-black);
	}
	.landing-terminal-bar {
		display: flex;
		align-items: center;
		gap: 1rem;
		padding: 0.5rem 1rem;
		border-bottom: 1px solid var(--sl-color-gray-5);
		font-size: 0.75rem;
		color: var(--sl-color-gray-4);
	}
	.landing-terminal-dots {
		display: flex;
		gap: 0.375rem;
	}
	.landing-terminal-dots span {
		width: 0.625rem;
		height: 0.625rem;
		border-radius: 50%;
		background: var(--sl-color-gray-5);
	}
	.landing-terminal pre {
		margin: 0;
		padding: 1.25rem 1.5rem;
		font-size: 0.875rem;
		line-height: 1.7;
	}
	.landing-section {
		margin-bottom: 4rem;
	}
	.landing-kicker {
		text-transform: uppercase;
		letter-spacing: 0.1em;
		font-size: 0.75rem;
		color: var(--sl-color-accent);
		margin-bottom: 0.25rem;
	}
	.landing-section h2 {
		font-size: 1.5rem;
		margin-bottom: 0.5rem;
	}
	.landing-section-intro {
		color: var(--sl-color-gray-3);
		max-width: 46rem;
		margin-bottom: 1.5rem;
	}
	.landing-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
		gap: 1rem;
	}
	.landing-card {
		border: 1px solid var(--sl-color-gray-5);
		border-radius: 0.75rem;
		padding: 1.25rem;
	}
	.landing-card-index {
		font-size: 0.75rem;
		color: var(--sl-color-gray-4);
	}
	.landing-card h3 {
		font-size: 1rem;
		margin: 0.25rem 0 0.5rem;
	}
	.landing-card p {
		font-size: 0.875rem;
		color: var(--sl-color-gray-3);
		margin: 0;
	}
	.kw { color: var(--sl-color-accent); }
	.str { color: var(--sl-color-green); }
</style>
