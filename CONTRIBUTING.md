# Contributing

Thank you for improving PaperAI.

## How to contribute

1. Create a focused branch for your change.
2. Keep changes small and reviewable.
3. Update documentation when behavior changes.
4. Test the local frontend and backend before opening a pull request.

## Suggested branch names

```text
feature/add-agent-name
fix/backend-route-error
docs/update-api-reference
```

## Code and content standards

- Keep agent instructions clear, reusable, and evidence-oriented.
- Do not add prompts that encourage unsupported claims.
- Do not add generated outputs as source files unless they are templates or examples.
- Keep local runtime outputs out of version control.

## Documentation standards

When adding a new agent, update:

- `SYSTEM_OVERVIEW.md`
- `README.md`
- Relevant workflow or template files
- Frontend agent selection if needed

## Pull request checklist

- [ ] The change has a clear purpose.
- [ ] Documentation is updated when needed.
- [ ] Local server still starts successfully.
- [ ] New agent rules avoid fabricated citations or evidence.
- [ ] Generated runtime files are not committed.
