# Articles

Four how-to articles on [dspy-agent-skills](..), written for AI and LLM enthusiasts who have heard of DSPy, GEPA, and RLM but haven't yet put them to work. Designed to ship across X.com (long-form articles), LinkedIn, and my Code and Context blog.

| # | Title | Focus | Est. read |
|---|---|---|---|
| [01](01-why-dspy-agent-skills.md) | Why I built dspy-agent-skills | Origin, the gap, the three ideas behind it | 6 min |
| [02](02-how-to-install-and-use.md) | How to install and use dspy-agent-skills | Install paths, first session, troubleshooting | 5 min |
| [03](03-inside-the-examples.md) | Inside the examples | Real GEPA improvement numbers, the saturation lesson | 9 min |
| [04](04-future-use-cases.md) | Where dspy-agent-skills goes next | Composition patterns, production, contribution | 7 min |

## Image placeholders

Each article ships inline `<!-- IMAGE PROMPT (Nano Banana): "..." -->` comments describing hero and diagram images. Generate with Gemini Nano Banana (or the image model of your choice) and replace the `placeholder-*.png` filenames with the final assets.

## References

All technical claims are sourced. Key citations used across the series:

- DSPy paper ([Khattab et al., ICLR 2024](https://arxiv.org/abs/2310.03714))
- GEPA paper ([Agrawal et al., ICLR 2026 Oral](https://arxiv.org/abs/2507.19457))
- Recursive Language Models ([Zhang, Kraska, Khattab, 2025](https://arxiv.org/abs/2512.24601))
- Alex Zhang's [RLM origin blog post](https://alexzhang13.github.io/blog/2025/rlm/)
- Simon Willison on [Agent Skills](https://simonwillison.net/2025/Dec/19/agent-skills/)
- Hamel Husain's [Field Guide to Rapidly Improving AI Products](https://hamel.dev/blog/posts/field-guide/) and ["Fuck You, Show Me The Prompt"](https://hamel.dev/blog/posts/prompt/)
- Arize's [GEPA vs. prompt-learning benchmark](https://arize.com/blog/gepa-vs-prompt-learning-benchmarking-different-prompt-optimization-approaches/)
- Agent Skills spec ([agentskills.io](https://agentskills.io/specification))
- Claude Code skills ([code.claude.com](https://code.claude.com/docs/en/skills))
- Codex CLI Agent Skills ([developers.openai.com/codex/skills](https://developers.openai.com/codex/skills))

## Writing notes

The articles were drafted and then humanized with the `writing-humanizer` skill — em dashes trimmed, filler removed, inline-header bullet lists converted to paragraphs, AI-prose patterns excised. If you spot a line that still reads as AI-generated, [open an issue](https://github.com/intertwine/dspy-agent-skills/issues).
