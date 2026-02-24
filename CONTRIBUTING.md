# Contributing to VidLens 🤝

Thank you for your interest in contributing! VidLens is community-driven.

## Ways to Contribute

- 🐛 Fix bugs
- 🔭 Add a new lens (new CV model)
- 📖 Improve documentation
- 🧪 Add tests
- 🎨 Improve the web UI

## Adding a New Lens (Most Wanted!)

1. Create `vidlens/lenses/your_lens.py`
2. Subclass `BaseLens` and implement `load_model()`, `process_frame()`, and optionally `train()`
3. Add to `vidlens/lenses/__init__.py` registry
4. Add tests in `tests/test_lenses.py`
5. Document in `docs/`

See existing lenses for examples.

## Development Setup

```bash
git clone https://github.com/your-org/vidlens
cd vidlens
pip install -e ".[all]"
pip install pytest ruff
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

We use `ruff` for linting:
```bash
ruff check vidlens/
ruff format vidlens/
```

## Pull Request Process

1. Fork → branch → commit → PR
2. Describe what you changed and why
3. Make sure tests pass
4. A maintainer will review within a few days

## Good First Issues

Look for issues labeled `good first issue` on GitHub.
These are specifically chosen to be beginner-friendly.
