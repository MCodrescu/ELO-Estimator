---
description: 'General code generation standards'
applyTo: '**'
---

# Code generation standards

## Simplicity
- Write straightforward, logically simple code. Avoid cleverness, indirection, and convolution.
- Add complexity only when a measured or clearly reasoned performance requirement, or an explicit correctness or security requirement, cannot be met by the simpler approach.
- Prefer the most readable solution that solves the actual problem — nothing more.

## Structure and modularity
- Break code into small, manageable, single-purpose pieces.
- Split logic across separate files/modules and import what you need rather than inlining everything.
- Avoid monoliths. Only keep code in one large unit when that genuinely fits the use case.
- Design isolated, independently testable components. Keep dependencies explicit and injectable where practical.
- Use config driven organization of settings and configuration. All common constants and settings shared across modules should be imported from a single location.
- Use pydantic models for data structures and model response templates.

## Style and naming
- Prefer object-oriented design. Use functional style only for stateless data transformations, pipeline operations, or when the language idiom strongly favors it (e.g., list comprehensions in Python).
- Use informative, descriptive names for variables, methods, and classes. No cryptic abbreviations.
- Write clear, informative docstrings in plain, human-sounding language.
  - No LLM-style filler or over-explaining (avoid phrases like "This function is responsible for...", "Let's...", "As we can see...").
  - State what it does, its inputs, outputs, and any important behavior — concisely.
- Do not add comments that merely restate the next line of code. Comment only to explain what the code cannot show on its own.
- Always include informative doc strings for every method including an args and returns sections with descriptions of each parameter.
- Separate code by newlines to make it more readable. Jumbled code is difficult to read. Code is read more than it is written. Add comments to describe intention when not clear explicitly from the method names.

## Imports (Python)
- Follow PEP 8. Group imports as: standard library, third-party, local — separated by blank lines.
- Sort imports alphabetically within each group.
- Keep imports at module scope unless a local import is required to avoid a cycle.
- For non-Python languages, follow the idiomatic import/module organization convention for that language (e.g., Go: stdlib then external then internal; TypeScript: external then internal then relative).


## Correctness and safety
- Never swallow exceptions. Catch only what you can handle, act on it, and re-raise or log with context otherwise.
- Catch specific exception types rather than broad `except:` clauses.
- Avoid race conditions. Use proper synchronization (locks, queues, atomic operations) around shared mutable state.
- Always keep security in mind: validate and sanitize inputs, avoid injection risks, handle secrets safely, and apply least privilege.
- Remove old code that is not connected or no longer utilized for any purpose.

## Performance
- Use parallelization or concurrency only after measurement or clear workload reasoning shows a likely performance bottleneck, and only when shared state can be managed safely.
- Start with the simplest correct approach; add concurrency only when evidence shows it will provide meaningful benefit.

## Ruff validation

- For Python changes, run `ruff check <touched-files>` after editing.
- Apply ruff changes automatically.
- In Agent mode, rerun Ruff after addressing issues.