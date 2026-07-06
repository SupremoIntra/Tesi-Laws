"""
Utility functions (console output).
"""
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
    HAS_RICH = True
except ImportError:
    class _FC:
        def print(self, *a, **kw): print(*[str(x) for x in a])
    console = _FC()
    HAS_RICH = False