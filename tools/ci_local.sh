#!/usr/bin/env bash
# Local continuous integration: the jobs of .github/workflows/ci.yml for the system of the
# development machine (Linux, macOS, or Windows with Git Bash), for when GitHub Actions is
# not available.
#
#   lint     ruff check and ruff format --check of src, tests, examples and docs
#   wheel    the wheel built once, installed with its optional dependencies in a virtual
#            environment per Python version
#   tests    the test suite against the installed wheel (from the tests directory, as
#            in CI), in parallel (pytest-xdist), then the command-line smoke test
#   docs     the documentation site built from docs/ (Sphinx), warnings as errors
#
# Usage: tools/ci_local.sh [PYTHON ...]   (default: 3.12 3.14)
# A PYTHON is a version or the path of an interpreter. With uv
# (https://docs.astral.sh/uv/), uv finds or installs the versions and makes the
# environments; without it, venv and pip make them, and a version is looked up as
# pythonX.Y on the PATH, then through the Windows launcher (py -X.Y). The environments
# live in .ci_local/ (or $CI_LOCAL_DIR). $CI_LOCAL_PACKAGES names further packages for the
# environments of the tests, such as "vtk gmsh": these independent readers then check the
# exported files (the tests skip those checks otherwise, as in the GitHub workflow).
#
# The Python versions run at the same time. The slow tests (marker slow: two thirds of the
# time, numerical checks that do not depend on the version of Python) run under one of
# them, $CI_LOCAL_SLOW (a version, by default the last one; "all": every version), the
# others take the fast tier (-m "not slow"). Each version gets its share of the cores, three
# times larger with the slow tests: $CI_LOCAL_WORKERS test processes per version (0: one
# process, without pytest-xdist) override it. An environment is kept from one run to the
# next while pyproject.toml and $CI_LOCAL_PACKAGES do not change, for a week at most (only
# the wheel is reinstalled, the disk caches of Numba of the package cleared);
# CI_LOCAL_FRESH=1 makes new ones.
# Exits with status 1 if a job fails; the log of each job is in .ci_local/<job>.log.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
work=${CI_LOCAL_DIR:-$root/.ci_local}
versions=("$@")
[ ${#versions[@]} -eq 0 ] && versions=(3.12 3.14)
read -r -a packages <<<"${CI_LOCAL_PACKAGES:-}"
mkdir -p "$work"
cd "$root"

case $(uname -s) in # where virtual environments keep their executables
    MINGW* | MSYS* | CYGWIN*) bin=Scripts ;;
    *) bin=bin ;;
esac
uv=$(command -v uv || true)

cores=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 2)

step() { printf '\n== %s ==\n' "$*"; }

run() {  # run JOB COMMAND...: the command's output goes to the log of the job
    local job=$1
    shift
    local start=$SECONDS
    if "$@" >"$work/$job.log" 2>&1; then
        printf '%-28s passed (%ds)\n' "$job" $((SECONDS - start))
    else
        printf '%-28s FAILED (%ds), log: %s\n' "$job" $((SECONDS - start)) "$work/$job.log"
        tail -n 40 "$work/$job.log"
        exit 1
    fi
}

native() {  # native PATH: the path as native programs read it (C:/... on Windows)
    if command -v cygpath >/dev/null; then cygpath -m "$1"; else printf '%s\n' "$1"; fi
}

interpreter() {  # interpreter PYTHON: the path of the interpreter (without uv)
    if [ -x "$1" ]; then
        printf '%s\n' "$1"
    elif command -v "python$1" >/dev/null; then
        command -v "python$1"
    elif command -v py >/dev/null && py "-$1" -c '' 2>/dev/null; then
        py "-$1" -c 'import sys; print(sys.executable)'
    else
        printf 'Python %s not found: install uv, or give the path of an interpreter\n' "$1" >&2
        return 1
    fi
}

label() {  # label PYTHON: the version of an interpreter given by its path, else PYTHON
    if [ -x "$1" ]; then
        "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])'
    else
        printf '%s\n' "$1"
    fi
}

new_env() {  # new_env PYTHON DIR: a virtual environment
    if [ -n "$uv" ]; then
        uv venv -q --allow-existing --python "$1" "$2"
    else
        local python
        python=$(interpreter "$1")
        "$python" -m venv "$2"
        "$2/$bin/python" -m pip install -q --upgrade pip
    fi
}

install() {  # install DIR REQUIREMENT...: packages installed in an environment
    local env=$1
    shift
    if [ -n "$uv" ]; then
        uv pip install --python "$env/$bin/python" "$@"
    else
        "$env/$bin/python" -m pip install "$@"
    fi
}

fingerprint() {  # fingerprint WORDS...: what an environment is made from
    "$work/lint/$bin/python" - "$@" ${packages[@]+"${packages[@]}"} <<'EOF'
import hashlib
import sys

digest = hashlib.sha256(open("pyproject.toml", "rb").read())
digest.update(" ".join(sys.argv[1:]).encode())
print(digest.hexdigest())
EOF
}

environment() {  # environment JOB PYTHON DIR EXTRAS: an environment with the wheel
    local job=$1 version=$2 env=$3 extras=$4 stamp
    stamp=$(fingerprint "$job" "$version" "$extras")
    if [ -z "${CI_LOCAL_FRESH:-}" ] && [ "$(cat "$env/ci-stamp" 2>/dev/null)" = "$stamp" ] \
        && [ -n "$(find "$env/ci-stamp" -mtime -7 2>/dev/null)" ]; then
        run "update-$job" install "$env" --force-reinstall --no-deps "$wheel"
        # Numba keys its disk cache on the contents of a source file, not on its date: a
        # kernel whose file did not change would keep the compiled code of the kernels it
        # calls in files that did (porous.py from behaviours.py and finite.py)
        find "$env" -path "*morphogen*" \( -name "*.nbi" -o -name "*.nbc" \) -delete
    else
        rm -rf "$env"
        new_env "$version" "$env"
        run "install-$job" install "$env" "$wheel[$extras]" ${packages[@]+"${packages[@]}"}
        printf '%s\n' "$stamp" >"$env/ci-stamp"
    fi
}

weight() {  # weight NAME: the share of the work of a Python version (3 with the slow tests)
    if [ "$slow" = all ] || [ "$1" = "$slow" ]; then echo 3; else echo 1; fi
}

tests() {  # tests PYTHON: the jobs of one Python version
    local version=$1 name env smoke share workers threads tier="all the tests"
    local parallel=() select=()
    name=$(label "$version")
    share=$((cores * $(weight "$name") / total))
    [ "$share" -gt 0 ] || share=1
    workers=${CI_LOCAL_WORKERS:-$share}
    # the threads of each test process (compiled kernels, SciPy, BLAS)
    threads=$((workers > 0 ? (share / workers > 0 ? share / workers : 1) : share))
    [ "$workers" -gt 0 ] && parallel=(-n "$workers" --dist loadfile)
    if [ "$(weight "$name")" -eq 1 ]; then
        tier="the fast tier"
        select=(-m "not slow")
    fi
    step "tests, Python $name: $tier, $workers processes of $threads threads"
    env="$work/py$name"
    environment "$name" "$version" "$env" test,surface,io,plot,mesh
    (
        cd tests
        export MORPHOGEN_TEST_THREADS=$threads OMP_NUM_THREADS=$threads \
            OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
        run "tests-$name" "$env/$bin/python" -m pytest -q -p no:cacheprovider --rootdir .. \
            -c ../pyproject.toml ${parallel[@]+"${parallel[@]}"} ${select[@]+"${select[@]}"} .
    )
    smoke="$work/smoke-$name"
    rm -rf "$smoke"
    mkdir -p "$smoke"
    (
        cd "$smoke"
        export PATH="$env/$bin:$PATH"
        run "smoke-$name" bash -c 'morphogen info && morphogen template boolean_spheres \
            -o smoke.yaml && morphogen run smoke.yaml -o smoke_out -q && \
            test -f smoke_out/results.csv && \
            python -c "from morphogen.mesh import mmg_executable; print(mmg_executable())"'
    )
}

docs() {  # docs PYTHON: the documentation site, warnings as errors
    step "docs"
    environment docs "$1" "$work/docs" docs
    rm -rf "$work/site"
    run sphinx "$work/docs/$bin/python" -m sphinx -b html -W --keep-going -q docs \
        "$work/site"
}

step "lint"
if [ ! -x "$work/lint/$bin/python" ]; then
    new_env "${versions[${#versions[@]} - 1]}" "$work/lint"
fi
install "$work/lint" -q ruff
run lint "$work/lint/$bin/ruff" check src tests examples docs
run format "$work/lint/$bin/ruff" format --check src tests examples docs

step "wheel"
rm -rf "$work/dist"
if [ -n "$uv" ]; then
    run build uv build --wheel --out-dir "$work/dist"
else
    run build "$work/lint/$bin/python" -m pip wheel --no-deps --wheel-dir "$work/dist" .
fi
# Native path: the extras keep MSYS from converting the argument on Windows.
wheel=$(native "$(ls "$work"/dist/*.whl)")

slow=${CI_LOCAL_SLOW:-$(label "${versions[${#versions[@]} - 1]}")}
total=0
for version in "${versions[@]}"; do
    total=$((total + $(weight "$(label "$version")")))
done
# The versions at the same time, and the documentation with the last one; the output of
# each job comes once they have all finished, in order.
outputs=()
pids=()
for version in "${versions[@]}"; do
    outputs+=("$work/jobs-$(label "$version").txt")
    tests "$version" >"$work/jobs-$(label "$version").txt" 2>&1 &
    pids+=($!)
done
outputs+=("$work/jobs-docs.txt")
docs "${versions[${#versions[@]} - 1]}" >"$work/jobs-docs.txt" 2>&1 &
pids+=($!)
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
cat "${outputs[@]}"
[ $status -eq 0 ] || exit 1

step "all jobs passed"
