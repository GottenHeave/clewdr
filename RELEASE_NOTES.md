# Release Notes

## What's New
- Use native incognito mode (`is_temporary`) for non-preserved chats (PR #145 by @GottenHeave)
- Web usage endpoint for enterprise accounts
- Update TLS emulation to Chrome 145

## Bug Fixes
- Fix `clewdr.toml` file permissions on Unix: now created with `0600` instead of default umask (#122)
- Add fallback for unknown `ContentBlock` types to prevent 422 deserialization errors (#97)
- Fix OAI `ImageUrl` to Claude `Image` format conversion in Claude Code proxy (PR #121 by @DragonFSKY)
- Fix enterprise usage tracking (PR #144 by @GottenHeave)
- Work around a bug in `tower-serve-static` (#147)

## Improvements
- Unify HTTP client construction across codebase
- Always use mimalloc as default allocator
- Unpin `tracing-subscriber`, allow ANSI color output
- Update dependencies to latest versions
- Use distroless Docker image
