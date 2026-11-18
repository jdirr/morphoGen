# API reference

The Python interface of morphoGen, written from its docstrings. Each package lists its
public names, those of its `__all__` (`from morphogen.homogenization import Material`),
linked to the module that defines them, then its modules; each module documents what it
defines. The command line is described in the [guide](guide.md#command-line), the keys
of the configuration files in the [architecture](architecture.md). The graphical
interface (`morphogen.gui`), the compiled kernels (`morphogen.kernels`) and the
functions behind the commands (`morphogen.cli`) are not part of it. Which of these names
stay stable from one version to the next is the subject of [](stability.md).

```{toctree}
:maxdepth: 2

api/morphogen
```
