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

### `0001-json-object-implies-an-object-schema.patch`

`response_format: {"type": "json_object"}` with no schema — the ordinary way
every OpenAI-compatible client asks for JSON — was silently ignored, and callers
only found out when parsing failed.

`oaicompat_chat_params_parse` defaults the missing schema to an empty object
(`tools/server/server-common.cpp:944`). Every downstream check then decides
whether a response format was requested by testing that the schema is
*non-empty* — `common/chat-auto-parser-generator.cpp:81` and `:134`,
`common/chat.cpp:1019` and `:2431` — so `{}` is indistinguishable from "no
constraint was asked for" and no grammar is built. The handlers that spell the
same test as `!is_null() && is_object()` (`chat.cpp:1170`, `:1318`, `:1707`,
`:1875`) are unaffected, which is why this reproduced on some chat formats and
not others. A top-level `json_schema`, or `response_format` of type
`json_schema`, produces a non-empty object and always worked.

The patch defaults the schema to `{"type": "object"}` instead, which is both
what makes the check pass and what the OpenAI contract means: the reply must be
a JSON object.

**Except when the caller supplied a `grammar`.** A schema beats a grammar
downstream, so manufacturing one for a request that already carries a grammar
silently discards what the caller asked for — a regression the first version of
this patch introduced and which cost a rebuild. Leaving the schema empty in that
case preserves the stock behaviour, where the grammar wins. Measured with
`grammar: 'root ::= "YES"'`:

| request | reply |
| --- | --- |
| grammar alone | `YES` |
| grammar + bare `json_object` | `YES` |
| grammar + an explicit `schema` | a JSON object — the schema wins |

That last row is stock llama.cpp behaviour, not something this patch changes: an
explicitly supplied schema has always taken precedence over a grammar.

Measured on this stack against Qwen3.6-27B at temperature 0, asking for one
plain English sentence *while* demanding JSON:

| | reply |
| --- | --- |
| before | `Cats are independent and affectionate pets that enjoy napping…` |
| after | `{"sentence": "Cats are independent and playful creatures…"}` |

`scripts/llm-chat-proxy.py` performs the same normalization per request
(`_normalize_json_object_response_format`). That covers :8003/:8004/:8008/:8012
without a rebuild, but not :8007 or the router members behind it, which do not
pass through the proxy — see `docs/model-router.md`. The two produce an
identical request, so the shim stays harmless once the patch is built in.

This is a local patch for a private deployment. llama.cpp's `AGENTS.md` sets
out what upstream expects of a contribution and exempts private forks; sending
it upstream is a separate, human-authored decision.
