"""Compatibility filename: launch the same TUI, without a separate CLI flow."""
from launcher import main
from optimizer_io import configure_stdio


if __name__ == "__main__":
    configure_stdio()
    raise SystemExit(main())
