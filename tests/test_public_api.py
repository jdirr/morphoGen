"""The public interface of morphoGen (docs/stability.md) against the list kept in
public_api.txt: a public name removed, added or whose parameters change is a change of
the interface, made on purpose. After such a change,

    python tests/test_public_api.py

writes the list again from the installed package, and the change log says what changed.
"""

from pathlib import Path, PureWindowsPath

from morphogen._public import parameters, public_interface, public_modules

KEPT = Path(__file__).with_name("public_api.txt")


def test_public_interface_kept():
    kept = KEPT.read_text(encoding="utf-8").splitlines()
    now = public_interface()
    removed = sorted(set(kept) - set(now))
    added = sorted(set(now) - set(kept))
    assert not removed and not added, (
        "the public interface changed (docs/stability.md); if on purpose, run "
        "`python tests/test_public_api.py` and say so in CHANGELOG.md\n"
        + "".join(f"  - {line}\n" for line in removed)
        + "".join(f"  + {line}\n" for line in added)
    )


def test_parameters_written_without_annotations():
    def f(a: int, b, /, c=1.5, *args, d: str = "x", e=None, **options) -> None:
        pass

    def g(a, *, b=(1, 2)):
        pass

    out, other = Path("runs/out"), PureWindowsPath("runs\\other")

    def h(out=out, other=other):
        pass

    assert parameters(f) == "(a, b, /, c=1.5, *args, d='x', e=None, **options)"
    assert parameters(g) == "(a, *, b=(1, 2))"
    # paths the same on every system: the list kept on Windows holds on Linux and macOS
    assert parameters(h) == "(out=Path('runs/out'), other=Path('runs/other'))"
    names = [name for name, _ in public_modules()]
    assert "morphogen.homogenization.identification" in names
    assert not any(n.startswith(("morphogen.gui", "morphogen.cli", "morphogen._")) for n in names)


if __name__ == "__main__":
    KEPT.write_text("\n".join(public_interface()) + "\n", encoding="utf-8", newline="\n")
    print(f"{KEPT}: the public interface written")
