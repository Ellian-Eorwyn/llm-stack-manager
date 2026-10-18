# Local llama.cpp patches

`deps/llama.cpp` is a plain clone pinned to an exact commit in
`dependencies.json`, not a fork. Anything this stack needs changed about it
lives in `patches/llama.cpp/` and is re-applied by
`scripts/install-dependencies.py` on every install.

## How it works

A dependency may declare `"patches": [...]` — repo-relative paths, applied in
order. `apply_patches()` discards the checkout back to the pinned ref first and
then applies them fresh, which makes the step idempotent and means a patched
checkout being permanently dirty is not something the dirty-worktree guard has
to protect. The checkout is a build artefact; it is not somewhere to keep work,
and anything edited by hand in `deps/` is discarded on the next install.

Each patch is `git apply --check`ed before it is applied, so a patch that has
stopped applying fails naming the patch rather than as a compiler error several
hundred lines into a CUDA build. In practice that only happens when the `ref`
in `dependencies.json` is bumped: refresh the patch against the new revision, or
drop it if the change landed upstream.

**A patch only takes effect after a rebuild.** `update.sh --skip-deps` and
`llm-stack-manager update` both deliberately skip the llama.cpp build, so they
will not pick one up.

## Current patches

### `0001-an-implied-object-schema-leaves-a-grammar-alone.patch`

`response_format: {"type": "json_object"}` with no schema — the ordinary way
every OpenAI-compatible client asks for JSON — used to be silently ignored: the
missing schema defaulted to `{}`, and every downstream check read an empty
schema as "no constraint was asked for". This stack patched that from August;
upstream fixed it in `acecd5603` (2026-09-12) by turning any empty schema into
`{"type": "object"}`, which is in the v0.7.0 pin.

Upstream's fix is unconditional, though, and a schema beats a grammar
downstream, so a request carrying its own `grammar` alongside a bare
`json_object` loses the grammar. The stack's own patch had already been bitten
by exactly that (it cost a rebuild), and preserved the grammar. So the patch is
now that one condition on upstream's line: `grammar.empty() &&`. Expected, with
`grammar: 'root ::= "YES"'`:

| request | reply |
| --- | --- |
| grammar alone | `YES` |
| grammar + bare `json_object` | `YES` |
| grammar + an explicit `schema` | a JSON object — the schema wins |

The last row is stock llama.cpp behaviour: an explicitly supplied schema has
always taken precedence over a grammar.

`scripts/llm-chat-proxy.py` performs the same normalization per request
(`_normalize_json_object_response_format`). That covers :8003/:8004/:8008/:8012
without a rebuild, but not :8007 or the router members behind it, which do not
pass through the proxy — see `docs/model-router.md`. The two produce an
identical request, so the shim stays harmless once the patch is built in.

This is a local patch for a private deployment. llama.cpp's `AGENTS.md` sets
out what upstream expects of a contribution and exempts private forks; sending
it upstream is a separate, human-authored decision.
