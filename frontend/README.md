# PaperAI Web Workspace

The frontend is a no-build browser client served by the FastAPI application. It supports:

- Creating and selecting persistent paper projects
- Uploading and monitoring extracted research files
- Selecting full-paper or section-level workflows
- Model-backed and prompt-only execution
- Live run progress polling
- Human approval, feedback, and rejection at checkpoints
- Browsing every agent artifact
- Markdown, Word, and LaTeX downloads

Run the application from the repository root with `python backend/server.py`; do not open `index.html` directly because the workspace depends on `/api` routes.
