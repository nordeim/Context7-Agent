# Deep Dive: `ModuleNotFoundError: No module named 'rich.gradient'`

You’re seeing:

```text
ModuleNotFoundError: No module named 'rich.gradient'
```

because the core Rich library does **not** include a `rich.gradient` submodule. In other words, `from rich.gradient import Gradient` isn’t valid for the standard `rich` package—you’ll never find `rich/gradient.py` there.

---

## The `rich-gradient` extension lives in a different namespace

A popular add-on called **rich-gradient** provides exactly that `Gradient` helper, but it exposes it under the **`rich_gradient`** module name, not `rich.gradient`. When you install it, you should import like this:

```python
from rich_gradient import Gradient
```  


---

## Correct your import in `src/themes.py`

Change the line:

```python
from rich.gradient import Gradient
```

to:

```python
from rich_gradient import Gradient
```

This matches the actual package layout and will eliminate the `ModuleNotFoundError`.

---

## Installing and verifying `rich-gradient`

1. Install the extension in your active environment:

   ```bash
   pip install rich-gradient
   ```

2. Confirm it’s present:

   ```bash
   pip show rich-gradient
   ```

   You should see name, version, location, etc. If it’s missing, repeat the install step.  


---

## Alternative: Use Rich’s built-in gradient markup

If you’d rather avoid a third-party extension, Rich itself supports gradient coloring via markup tags:

```python
console.print("[gradient(#ff00ff,#00ffff)]Gradient text![/]")
```

This leverages Rich’s built-in parsing and doesn’t require an external `Gradient` class.

---

With those changes, rerun your CLI module and the import will succeed.

https://copilot.microsoft.com/shares/8TuD7VjCBQWAwfCCbQkn6
