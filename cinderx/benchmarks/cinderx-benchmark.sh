#!/usr/bin/env bash
###############################################################################
# cinderx-benchmark.sh
#
# Fully self-contained benchmark harness for CinderX (the Meta JIT/Static-Python
# extension) on a locally-built, optimized CPython.  Copy this one file to a
# fresh Linux box that has git + a C/C++20 toolchain + build essentials, run it,
# and it will:
#
#   1. Clone CPython and build it with PGO + LTO into a local prefix (the "plain"
#      CPython, with no CinderX linked in).
#   2. Build the CinderX static archives and relink CPython into a SECOND prefix
#      (the "static" CPython, with _cinderx as a builtin module). Only one PGO
#      build runs — the static binary is a `make python` relink of the same tree.
#   3. Create TWO benchmark virtualenvs (one per interpreter) and install
#      pyperformance + the fastmark benchmark dependencies in both:
#         - plain venv:  CinderX installed dynamically (PyPI wheel / source build)
#         - static venv: pure-Python cinderx package (PythonLib) via a .pth file
#   4. Generate per-benchmark JIT lists once (shared by both interpreters).
#   5. Run four benchmark suites, each across EIGHT configurations — the cross
#      product of {plain, static} CPython and four JIT modes (off / no-jit /
#      auto / per-benchmark jit-list):
#         1. plain-off       5. static-off
#         2. dyn-nojit       6. static-nojit
#         3. dyn-auto        7. static-auto
#         4. dyn-jitlist     8. static-jitlist
#      The four suites are: A. CinderX built-in lightweight benchmarks,
#      B. fastmark, C. Static-Python variants, D. pyperformance.
#   6. Emit an 8-way comparison table / markdown report for every suite.
#
# The eight configurations are driven entirely by environment variables that the
# embedded sitecustomize.py interprets:
#   CINDERX_DISABLE=1       -> don't import cinderx at all          (off configs)
#   CINDERX_NO_JIT=1        -> import cinderx but DON'T enable JIT  (no-jit configs)
#   BENCH_JITLIST_DIR=<dir> -> load the per-benchmark JIT list      (jitlist configs)
#   (none of the above)     -> cinderx.jit.auto()                   (auto configs)
#
# Everything the script needs at runtime (the JIT-list generator, the env-driven
# sitecustomize, and the report generator) is embedded below as heredocs, so
# there are no companion files to copy.
#
# This reproduces the methodology documented in:
#   ~/python3.14-cinderx/cinderx-benchmark-results.md
#   ~/python3.14-cinderx/pyperf-work/pyperformance-cinderx-results.md
#   ~/python3.14-cinderx/pyperf-work/jitlist-3way-results.md
#
# Run `bash cinderx-benchmark.sh --help` for all options.
###############################################################################

set -u -o pipefail

# ---------------------------------------------------------------------------
# Defaults (override via env var or CLI flag; CLI flags win).
# ---------------------------------------------------------------------------
CPYTHON_REPO="${CPYTHON_REPO:-https://github.com/python/cpython.git}"
CPYTHON_TAG="${CPYTHON_TAG:-v3.14.5}"          # branch or tag to build
CINDERX_REPO="${CINDERX_REPO:-https://github.com/facebookincubator/cinderx.git}"
CINDERX_TAG="${CINDERX_TAG:-main}"             # only used with --cinderx-source
CINDERX_VERSION="${CINDERX_VERSION:-}"         # pin a PyPI version, e.g. 2026.6.25.0 (empty = latest)
CINDERX_SOURCE="${CINDERX_SOURCE:-0}"          # 1 = build CinderX from git instead of PyPI wheel (for the dynamic/plain venv)
CINDERX_CC="${CINDERX_CC:-}"                    # override C compiler for CPython + CinderX (default: auto-detect); CLI: --cc
CINDERX_CXX="${CINDERX_CXX:-}"                  # override C++ compiler for CPython + CinderX (default: auto-detect); CLI: --cxx
# Where each compiler override came from, for validation error messages. Seeded
# from the env vars now; the CLI parser rewrites these to "--cc"/"--cxx" when the
# flags are used (CLI beats env). Empty => not explicitly provided => auto-detect.
CC_ORIGIN=""; [ -n "$CINDERX_CC" ] && CC_ORIGIN="env var CINDERX_CC"
CXX_ORIGIN=""; [ -n "$CINDERX_CXX" ] && CXX_ORIGIN="env var CINDERX_CXX"
CINDERX_LIBSTDCXX_A="${CINDERX_LIBSTDCXX_A:-}"  # override static libstdc++.a fallback path (default: ask the C++ compiler via -print-file-name)

WORKDIR="${WORKDIR:-}"                          # REQUIRED: root for everything. No default — pass --workdir DIR (or set $WORKDIR).
JOBS="${JOBS:-$(nproc)}"                       # make -j parallelism
AFFINITY="${AFFINITY:-}"                       # CPU core range for taskset, e.g. "8-11" (empty = no pinning)
TRIALS="${TRIALS:-3}"                          # repeats for built-in / static suites (min reported)
PYPERF_MODE="${PYPERF_MODE:---fast}"           # --fast or "" (empty = full steady-state)
REGEN_JITLISTS="${REGEN_JITLISTS:-0}"          # 1 = regenerate JIT lists even if cached
JIT_THRESHOLD="${JIT_THRESHOLD:-2}"            # gen_jitlist hot-function threshold (2=canonical)
JIT_BUDGET="${JIT_BUDGET:-3.0}"               # gen_jitlist workload budget seconds

# Which phases to run (all on by default). Disabled via --skip-* flags.
DO_BUILD_CPYTHON=1
DO_INSTALL_CINDERX=1
DO_VENV=1
DO_JITLISTS=1
RUN_BUILTIN=1
RUN_FASTMARK=1
RUN_STATIC=1
RUN_PYPERF=1

# pyperformance benchmark set for suite D (5 prior winners + 5 prior regressions:
# high-signal, fast enough for repeated runs). Override with --pyperf-benches.
PYPERF_BENCHES="${PYPERF_BENCHES:-richards,richards_super,spectral_norm,chaos,deltablue,fannkuch,raytrace,generators,go,nqueens}"

# fastmark work-scale factor (lower = faster; 100 is fastmark's default).
FASTMARK_SCALE="${FASTMARK_SCALE:-100}"

# BOLT post-link optimization (opt-in via --bolt). BOLT rewrites each interpreter
# binary's code layout from profile data (basic-block/function reordering, cold
# splitting), directly targeting the ±10% binary-layout effect the foobar7/8
# analysis measured between the plain and static builds. Off by default: it adds
# build time and needs llvm-bolt (+ merge-fdata). When enabled, CPython is linked
# with -Wl,--emit-relocs so BOLT can run in relocation mode (required for function
# reordering); the static relink inherits that flag via configure's LDFLAGS.
DO_BOLT="${DO_BOLT:-0}"
# Representative pyperformance benchmarks used to collect the BOLT profile — short
# but they exercise the interpreter-core hot paths. Override with --bolt-benches.
BOLT_BENCHES="${BOLT_BENCHES:-richards,raytrace,deltablue}"

# ---------------------------------------------------------------------------
# Derived paths.
# ---------------------------------------------------------------------------
SRC_CPYTHON="$WORKDIR/cpython"
SRC_CINDERX="$WORKDIR/cinderx"
PY_PREFIX="$WORKDIR/python-install"                # plain CPython (no CinderX linked in)
PY_PREFIX_STATIC="$WORKDIR/python-install-static"  # static CPython (builtin _cinderx)
VENV="$WORKDIR/venv"                               # plain venv (dynamic CinderX wheel)
VENV_STATIC="$WORKDIR/venv-static"                 # static venv (PythonLib via .pth)
RESULTS="$WORKDIR/results"
JITLIST_DIR="$WORKDIR/jitlists/lists"
HELPERS="$WORKDIR/helpers"
LOGDIR="$RESULTS/logs"
STATIC_BUILD_DIR="$WORKDIR/cinderx-static-build"   # cmake build tree for the CinderX .a archives

# Populated after the venvs exist.
VPY=""           # plain venv python
VPY_STATIC=""    # static venv python

# Populated by detect_toolchain(): a single C++20 toolchain used to build BOTH
# CPython (./configure CC=/CXX=) and the CinderX archives, so their libstdc++/ABI
# match. TOOLCHAIN_LIBSTDCXX is a static libstdc++.a kept only as a link fallback.
TOOLCHAIN_CC=""; TOOLCHAIN_CXX=""; TOOLCHAIN_LIBSTDCXX=""

# ---------------------------------------------------------------------------
# Pretty logging.
# ---------------------------------------------------------------------------
c_blue=$'\033[1;34m'; c_grn=$'\033[1;32m'; c_yel=$'\033[1;33m'; c_red=$'\033[1;31m'; c_rst=$'\033[0m'
[ -t 1 ] || { c_blue=; c_grn=; c_yel=; c_red=; c_rst=; }
log()  { printf '%s[%s]%s %s\n' "$c_blue" "$(date +%H:%M:%S)" "$c_rst" "$*"; }
ok()   { printf '%s  ✓ %s%s\n' "$c_grn" "$*" "$c_rst"; }
warn() { printf '%s  ! %s%s\n' "$c_yel" "$*" "$c_rst"; }
die()  { printf '%s  ✗ %s%s\n' "$c_red" "$*" "$c_rst" >&2; exit 1; }

# Wrap a command with taskset if AFFINITY is set (for direct shell invocation).
pin() { if [ -n "$AFFINITY" ]; then taskset -c "$AFFINITY" "$@"; else "$@"; fi; }

# Taskset prefix as a plain token list, for use after `env` (which cannot exec a
# shell function). Empty when no affinity requested.
taskset_prefix() { if [ -n "$AFFINITY" ]; then printf 'taskset -c %s' "$AFFINITY"; fi; }

###############################################################################
# --help
###############################################################################
usage() {
  cat <<'USAGE'
cinderx-benchmark.sh — build plain + static CPython and benchmark CinderX
across EIGHT configurations (static/dynamic linking × 4 JIT modes).

USAGE:
  bash cinderx-benchmark.sh --workdir DIR [OPTIONS]

  --workdir is REQUIRED (there is no default work directory). It runs the entire
  pipeline: build the plain CPython (PGO+LTO),
  relink a second "static" CPython with _cinderx built in, create two benchmark
  venvs (dynamic CinderX wheel + static PythonLib), generate JIT lists, then run
  all four suites across all eight configurations and write reports.

THE EIGHT CONFIGURATIONS (every suite runs each one):
  1. plain-off        plain CPython, CinderX not imported  (CINDERX_DISABLE=1)
  2. dyn-nojit        plain CPython, dynamic CinderX imported, JIT not enabled
  3. dyn-auto         plain CPython, dynamic CinderX, cinderx.jit.auto()
  4. dyn-jitlist      plain CPython, dynamic CinderX, per-benchmark JIT lists
  5. static-off       static CPython (builtin _cinderx), not imported
  6. static-nojit     static CPython, CinderX imported, JIT not enabled
  7. static-auto      static CPython, cinderx.jit.auto()
  8. static-jitlist   static CPython, per-benchmark JIT lists

PHASE CONTROL (skip expensive phases to iterate; phases are also auto-skipped
if their output already exists):
  --skip-build-cpython   Reuse existing CPython installs (plain + static).
  --skip-cinderx         Don't (re)install the dynamic CinderX into the plain venv.
  --skip-venv            Reuse existing benchmark venvs (plain + static).
  --skip-jitlists        Don't generate JIT lists (jitlist configs fall back to cached).
  --skip-builtin         Skip suite A (built-in lightweight benchmarks).
  --skip-fastmark        Skip suite B (fastmark / pyperformance via cinderx).
  --skip-static          Skip suite C (Static-Python variants).
  --skip-pyperf          Skip suite D (pyperformance).
  --only-bench           Shorthand: skip all build/install/venv phases, just
                         run the benchmark suites against existing installs.
  --regen-jitlists       Force regeneration of JIT lists even if cached.

SOURCES / VERSIONS:
  --cpython-tag TAG      CPython git tag/branch to build      (default: v3.14.5)
  --cpython-repo URL     CPython git remote                   (default: github.com/python/cpython)
  --cinderx-version VER  Pin a CinderX PyPI version (plain venv wheel) (default: latest)
  --cinderx-source       Build the dynamic CinderX from a git checkout instead of PyPI.
  --cinderx-repo URL     CinderX git remote (used for both static build and source build).
  --cinderx-tag TAG      CinderX git tag/branch.

COMPILER / TOOLCHAIN:
  --cc PATH              C compiler for CPython + CinderX (full path or command
                        name, e.g. --cc /opt/gcc-15/bin/gcc). Overrides CINDERX_CC.
  --cxx PATH            C++ compiler for CPython + CinderX (full path or command
                        name, e.g. --cxx /opt/gcc-15/bin/g++). Overrides CINDERX_CXX.
                        Precedence: --cc/--cxx > CINDERX_CC/CINDERX_CXX > auto-detect.
                        When a compiler is given explicitly, auto-detection is
                        skipped for it and the path is validated (must exist and be
                        executable). Giving only --cxx derives the sibling C
                        compiler (g++->gcc, clang++->clang) unless --cc is also set.

RUN TUNING:
  --workdir DIR          Root for sources/build/venv/results (REQUIRED — no default)
  --jobs N               make -j parallelism                  (default: nproc)
  --affinity RANGE       Pin benchmark processes, e.g. "8-11" (default: none)
  --trials N             Repeats for built-in/static suites   (default: 3)
  --pyperf-mode MODE     "--fast" or "" for steady-state      (default: --fast)
  --pyperf-benches LIST  Comma-separated pyperformance set for suite D.
  --fastmark-scale N     fastmark work scale (lower=faster)   (default: 100)
  --jit-threshold N      gen_jitlist hot threshold            (default: 2)
  --jit-budget SECS      gen_jitlist workload budget          (default: 3.0)

BOLT POST-LINK OPTIMIZATION (optional):
  --bolt                 After building each interpreter, run LLVM BOLT to rewrite
                        its code layout from a profile (basic-block + function
                        reordering, cold-code splitting). This directly targets the
                        ±10% binary-layout effect measured between the plain and
                        static builds. OFF by default (adds build time; needs
                        llvm-bolt + merge-fdata on PATH). When enabled, CPython is
                        linked with -Wl,--emit-relocs so BOLT can reorder functions
                        (relocation mode); the static relink inherits that flag. If
                        llvm-bolt is absent, or a reused binary lacks relocations,
                        BOLT is skipped with a warning (the build still completes).
  --no-bolt              Explicitly disable BOLT (the default).
  --bolt-benches LIST    Comma-separated pyperformance benchmarks used to collect
                        the BOLT profile (default: richards,raytrace,deltablue).

  -h, --help             Show this help and exit.

ENVIRONMENT VARIABLES:
  Every option above has a matching env var (CPYTHON_TAG, CINDERX_VERSION,
  WORKDIR, JOBS, AFFINITY, TRIALS, PYPERF_MODE, PYPERF_BENCHES, FASTMARK_SCALE,
  JIT_THRESHOLD, JIT_BUDGET, DO_BOLT, BOLT_BENCHES, ...). CLI flags take
  precedence over env vars.
  WORKDIR is required: set it via --workdir or the $WORKDIR env var (no default).
  CINDERX_CC / CINDERX_CXX override the auto-detected compiler used for BOTH
  CPython and the CinderX archives (the --cc / --cxx flags take precedence over
  them); CINDERX_LIBSTDCXX_A overrides the static libstdc++.a fallback path used
  when relinking the static interpreter.

OUTPUT:
  Results, logs and markdown reports are written under  <workdir>/results/ .
  A top-level SUMMARY.md links all per-suite 8-way reports.

REQUIREMENTS (must be pre-installed on a fresh box):
  git, make, a C++20 compiler (gcc 13+ or clang 18+), cmake + ninja + ar (the
  static interpreter is always built), and the usual CPython build deps (zlib,
  openssl/libssl, libffi, readline, bzip2, lzma, sqlite headers).

EXAMPLES:
  bash cinderx-benchmark.sh --workdir ~/cinderx-bench
  bash cinderx-benchmark.sh --workdir ~/cinderx-bench --affinity 8-11 --trials 5
  bash cinderx-benchmark.sh --workdir ~/cinderx-bench --only-bench --skip-jitlists
  bash cinderx-benchmark.sh --workdir ~/cinderx-bench --bolt   # BOLT both binaries
  CPYTHON_TAG=v3.14.5 bash cinderx-benchmark.sh --workdir ~/cinderx-bench --cinderx-version 2026.6.25.0
  bash cinderx-benchmark.sh --workdir ~/cinderx-bench --cinderx-source --cinderx-tag main
  WORKDIR=~/cinderx-bench bash cinderx-benchmark.sh --affinity 8-11 --skip-fastmark
USAGE
}

###############################################################################
# Argument parsing.
###############################################################################
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-build-cpython) DO_BUILD_CPYTHON=0 ;;
    --skip-cinderx)       DO_INSTALL_CINDERX=0 ;;
    --skip-venv)          DO_VENV=0 ;;
    --skip-jitlists)      DO_JITLISTS=0 ;;
    --skip-builtin)       RUN_BUILTIN=0 ;;
    --skip-fastmark)      RUN_FASTMARK=0 ;;
    --skip-static)        RUN_STATIC=0 ;;
    --skip-pyperf)        RUN_PYPERF=0 ;;
    --only-bench)         DO_BUILD_CPYTHON=0; DO_INSTALL_CINDERX=0; DO_VENV=0 ;;
    --regen-jitlists)     REGEN_JITLISTS=1 ;;
    --bolt)               DO_BOLT=1 ;;
    --no-bolt)            DO_BOLT=0 ;;
    --bolt-benches)       BOLT_BENCHES="$2"; shift ;;
    --cinderx-source)     CINDERX_SOURCE=1 ;;
    --cpython-tag)        CPYTHON_TAG="$2"; shift ;;
    --cpython-repo)       CPYTHON_REPO="$2"; shift ;;
    --cinderx-version)    CINDERX_VERSION="$2"; shift ;;
    --cinderx-repo)       CINDERX_REPO="$2"; shift ;;
    --cinderx-tag)        CINDERX_TAG="$2"; shift ;;
    --cc)                 CINDERX_CC="$2";  CC_ORIGIN="--cc";  shift ;;
    --cxx)                CINDERX_CXX="$2"; CXX_ORIGIN="--cxx"; shift ;;
    --workdir)            WORKDIR="$2"; shift ;;
    --jobs)               JOBS="$2"; shift ;;
    --affinity)           AFFINITY="$2"; shift ;;
    --trials)             TRIALS="$2"; shift ;;
    --pyperf-mode)        PYPERF_MODE="$2"; shift ;;
    --pyperf-benches)     PYPERF_BENCHES="$2"; shift ;;
    --fastmark-scale)     FASTMARK_SCALE="$2"; shift ;;
    --jit-threshold)      JIT_THRESHOLD="$2"; shift ;;
    --jit-budget)         JIT_BUDGET="$2"; shift ;;
    -h|--help)            usage; exit 0 ;;
    *) die "Unknown option: $1  (try --help)" ;;
  esac
  shift
done

# --workdir is required (no default): a benchmark run creates large source/build
# trees, so we never pick a directory on the user's behalf.
if [ -z "$WORKDIR" ]; then
  die "--workdir is required (no default). Pass --workdir DIR (or set \$WORKDIR) to choose the
    root for sources/build/venv/results, e.g.  --workdir ~/cinderx-bench . See --help."
fi

# Re-derive paths in case --workdir changed them.
SRC_CPYTHON="$WORKDIR/cpython"
SRC_CINDERX="$WORKDIR/cinderx"
PY_PREFIX="$WORKDIR/python-install"
PY_PREFIX_STATIC="$WORKDIR/python-install-static"
VENV="$WORKDIR/venv"
VENV_STATIC="$WORKDIR/venv-static"
RESULTS="$WORKDIR/results"
JITLIST_DIR="$WORKDIR/jitlists/lists"
HELPERS="$WORKDIR/helpers"
LOGDIR="$RESULTS/logs"
STATIC_BUILD_DIR="$WORKDIR/cinderx-static-build"

###############################################################################
# The eight benchmark configurations.
#
# Each entry is  name|venv_kind|env_assignments  where:
#   - venv_kind is "plain" (dynamic CinderX wheel) or "static" (builtin _cinderx).
#   - env_assignments is a (possibly empty) space-separated list of VAR=VALUE
#     tokens passed through `env` to the benchmark process; the embedded
#     sitecustomize.py interprets them to pick the CinderX/JIT policy.
# Empty env => default => cinderx.jit.auto().
###############################################################################
CONFIG_ORDER="plain-off,dyn-nojit,dyn-auto,dyn-jitlist,static-off,static-nojit,static-auto,static-jitlist"
ALL_CONFIGS=(
  "plain-off|plain|CINDERX_DISABLE=1"
  "dyn-nojit|plain|CINDERX_NO_JIT=1"
  "dyn-auto|plain|"
  "dyn-jitlist|plain|BENCH_JITLIST_DIR=$JITLIST_DIR"
  "static-off|static|CINDERX_DISABLE=1"
  "static-nojit|static|CINDERX_NO_JIT=1"
  "static-auto|static|"
  "static-jitlist|static|BENCH_JITLIST_DIR=$JITLIST_DIR"
)

# Resolve a config's venv-kind to its venv python interpreter.
config_vpy() {
  case "$1" in
    plain)  printf '%s\n' "$VENV/bin/python" ;;
    static) printf '%s\n' "$VENV_STATIC/bin/python" ;;
    *)      return 1 ;;
  esac
}

###############################################################################
# Preflight: required tools.
###############################################################################
preflight() {
  log "Preflight: checking required tools"
  local missing=0
  for t in git make; do
    command -v "$t" >/dev/null 2>&1 || { warn "missing required tool: $t"; missing=1; }
  done
  # A C compiler must be present. We only check for existence here — do NOT report
  # a version from it, because the system gcc/clang seen on PATH is often NOT the
  # compiler the build uses: detect_toolchain() (called below) selects a C++20
  # toolchain that may be a different compiler entirely (e.g. clang++ 22 instead of
  # gcc 11.5). The authoritative CC/CXX version report is printed after detection.
  if ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
    warn "no C compiler (gcc/clang) found"; missing=1
  fi
  command -v taskset >/dev/null 2>&1 || [ -z "$AFFINITY" ] || \
    warn "taskset not found but --affinity set; pinning will be skipped"
  command -v /usr/bin/time >/dev/null 2>&1 || warn "/usr/bin/time not found; built-in/static timing may be less precise"

  # The static interpreter is ALWAYS built now, so cmake + ninja + ar are required
  # to build the CinderX archives and relink CPython.
  for t in cmake ninja ar; do
    command -v "$t" >/dev/null 2>&1 || { warn "missing required tool: $t (needed to build the static CPython)"; missing=1; }
  done

  # Detect ONE C++20 toolchain to build both CPython and the CinderX archives
  # with, so their libstdc++/ABI match. A C++20 compiler is now mandatory because
  # the static interpreter (builtin _cinderx) is always built.
  detect_toolchain
  if [ -n "$TOOLCHAIN_CXX" ]; then
    # Report BOTH compilers that detect_toolchain() actually selected — these are
    # exactly the CC/CXX passed to ./configure and cmake (and already reflect any
    # --cc/--cxx or CINDERX_CC/CINDERX_CXX overrides). Show each one's own
    # --version so the preflight can't misreport the system gcc.
    local _cc="${TOOLCHAIN_CC:-cc}"
    ok "Toolchain (CPython + CinderX): CC=$_cc  CXX=$TOOLCHAIN_CXX"
    ok "  CC  version: $("$_cc" --version 2>&1 | head -1)"
    ok "  CXX version: $("$TOOLCHAIN_CXX" --version 2>&1 | head -1)"
    if [ -n "$TOOLCHAIN_LIBSTDCXX" ]; then
      ok "Static libstdc++ fallback present: $TOOLCHAIN_LIBSTDCXX (used only if the default dynamic -lstdc++ link fails)"
    else
      warn "No static libstdc++.a found; the static link will use dynamic -lstdc++ only (matching compiler should make this safe)"
    fi
  else
    warn "No suitable C++20 compiler (gcc 13+ or clang) found; required to build the static CPython"; missing=1
  fi

  # BOLT (opt-in): check llvm-bolt availability. If it is missing we DISABLE BOLT
  # here (rather than failing) so the rest of the pipeline still runs — and so
  # build_cpython does NOT add -Wl,--emit-relocs (which would otherwise bloat the
  # binary for a BOLT pass that can never happen). Preflight runs before the build,
  # so this decision propagates correctly.
  if [ "$DO_BOLT" -eq 1 ]; then
    if command -v llvm-bolt >/dev/null 2>&1; then
      ok "BOLT enabled: llvm-bolt found ($(llvm-bolt --version 2>&1 | sed -n 's/.*LLVM version/LLVM/p;q'))"
      if ! command -v merge-fdata >/dev/null 2>&1; then
        warn "merge-fdata not found (ships with LLVM BOLT); multi-benchmark BOLT profiles cannot be merged — a single profile will be used instead."
      fi
      if ! command -v readelf >/dev/null 2>&1 && ! command -v llvm-readelf >/dev/null 2>&1; then
        warn "readelf/llvm-readelf not found; cannot verify -Wl,--emit-relocs before BOLT (BOLT will be attempted regardless)."
      fi
    else
      warn "llvm-bolt not found on PATH; --bolt requested but BOLT will be SKIPPED. Install LLVM BOLT (e.g. distro 'bolt'/'llvm' package) and re-run."
      DO_BOLT=0
    fi
  fi

  [ "$missing" -eq 0 ] || die "Missing prerequisites above. Install them and re-run."
  ok "All required tools present"
  mkdir -p "$WORKDIR" "$RESULTS" "$LOGDIR" "$HELPERS"
}

###############################################################################
# Write the embedded Python helpers to disk.
###############################################################################
write_helpers() {
  log "Writing embedded helpers to $HELPERS"

  # --- gen_jitlist.py : per-benchmark JIT-list generator ------------------
  cat > "$HELPERS/gen_jitlist.py" <<'PYEOF'
#!/usr/bin/env python3
# Generate a per-benchmark CinderX JIT list by exercising a pyperformance
# benchmark's workload under a low JIT threshold, then dumping the functions
# the JIT actually compiled (the genuinely hot ones), one module:qualname/line.
#
# Usage:  python gen_jitlist.py <run_benchmark.py> [--budget SECS] [-- <bench args>]
# Env:    JIT_THRESHOLD (default 50) = hot-function call threshold for the short run.
#
# cProfile only records co_name, losing class/module, so it can't build a
# reliable module:qualname list. cinderx.jit.get_compiled_functions() returns
# real function objects, so __module__ + __qualname__ give the exact format.
import sys
import time
import functools
import runpy

import pyperf

_BUDGET = 2.0
argv = sys.argv[1:]
if "--budget" in argv:
    i = argv.index("--budget")
    _BUDGET = float(argv[i + 1])
    del argv[i : i + 2]
bench_path = argv[0]
bench_args = argv[1:]

captured = []  # (kind, name, callable, args)


class CaptureRunner(pyperf.Runner):
    def bench_func(self, name, func, *args, **kwargs):
        captured.append(("func", name, func, args))
        return None

    def bench_time_func(self, name, time_func, *args, **kwargs):
        captured.append(("time", name, time_func, args))
        return None

    def bench_async_func(self, name, func, *args, **kwargs):
        captured.append(("async", name, func, args))
        return None


pyperf.Runner = CaptureRunner

# Drive the benchmark's __main__ with a controlled argv so its parse_args works.
sys.argv = [bench_path, "--worker", "-l", "1", "-w", "0", "-n", "1"] + bench_args
try:
    runpy.run_path(bench_path, run_name="__main__")
except SystemExit:
    pass

import cinderx.jit as jit  # noqa: E402

if not jit.is_enabled():
    print("ERROR: cinderx JIT not enabled", file=sys.stderr)
    sys.exit(2)

# Low threshold on the short generation run approximates auto()'s threshold on a
# full run, and captures functions called few times but doing heavy internal work
# (nbody.advance, fannkuch, meteor.solve) that threshold 1000 would miss.
import os  # noqa: E402

_THRESHOLD = int(os.environ.get("JIT_THRESHOLD", "50"))
jit.compile_after_n_calls(_THRESHOLD)


def _ids(funcs):
    return {id(f): f for f in funcs}


def run_workload():
    if not captured:
        print("ERROR: no workload captured", file=sys.stderr)
        sys.exit(3)
    deadline = time.perf_counter() + _BUDGET
    reps = 0
    while True:
        for kind, name, func, args in captured:
            try:
                if kind == "time":
                    func(64, *args)
                elif kind == "async":
                    import asyncio

                    asyncio.run(func(*args))
                else:
                    f = functools.partial(func, *args) if args else func
                    f()
            except Exception as e:
                print(f"WARN: workload {name!r} raised {e!r}", file=sys.stderr)
        reps += 1
        if time.perf_counter() >= deadline and reps >= 2:
            break
    return reps


before = _ids(jit.get_compiled_functions())
reps = run_workload()
after = _ids(jit.get_compiled_functions())

new_ids = set(after) - set(before)
entries = set()
for fid in new_ids:
    fn = after[fid]
    mod = getattr(fn, "__module__", None)
    qual = getattr(fn, "__qualname__", None)
    if mod and qual:
        entries.add(f"{mod}:{qual}")

print(
    f"INFO: reps={reps} budget={_BUDGET}s compiled_before={len(before)} "
    f"compiled_after={len(after)} new={len(new_ids)} emitted={len(entries)}",
    file=sys.stderr,
)
for e in sorted(entries):
    print(e)
PYEOF

  # --- sitecustomize.py : env-driven CinderX policy -----------------------
  cat > "$HELPERS/sitecustomize.py" <<'PYEOF'
# Env-driven CinderX policy for the 8-way comparison. Exactly one mode applies,
# checked in priority order:
#   (a) CINDERX_DISABLE=1       -> CinderX off entirely; don't import cinderx
#                                  at all (the "off" configurations).
#   (b) CINDERX_NO_JIT=1        -> import cinderx (init the runtime / Static Python
#                                  machinery) but DON'T enable the JIT: no auto(),
#                                  no jit list (the "no-jit" configurations).
#   (c) BENCH_JITLIST_DIR=<dir> -> import cinderx and load the per-benchmark
#                                  <name>.jitlist, NO auto() (jit list gates which
#                                  funcs compile; with no threshold they compile on
#                                  first call -- validated in the profiling task).
#   (d) (none of the above)     -> cinderx.jit.auto()  (blanket auto-JIT baseline).
# Benchmark name is detected from sys.argv[0] = .../bm_<name>/run_benchmark.py.
#
# IMPORTANT (pyperformance workers): pyperformance runs each benchmark inside an
# isolated "compat" venv that contains neither this file nor cinderx. We are put
# back on that worker's path by placing a *minimal* directory holding a copy of
# this file on PYTHONPATH (whitelisted by both pyperformance and pyperf, so it
# survives into the worker). cinderx itself is made importable there by
# _add_cinderx_site() below, which site.addsitedir()'s the real benchmark venv's
# site-packages -- APPENDING it (so it lands after the stdlib and after the
# compat venv's own packages: no shadowing) and PROCESSING its .pth files (needed
# for the static config, whose cinderx comes from cinderx_pythonlib.pth).
import os
import sys


def _add_cinderx_site():
    # Make cinderx importable inside pyperformance's isolated compat venv without
    # shadowing anything. CINDERX_SITE_DIR is an os.pathsep-separated list of the
    # config's real venv site-packages director(ies).
    raw = os.environ.get("CINDERX_SITE_DIR", "")
    if not raw:
        return
    import site
    for d in raw.split(os.pathsep):
        if d and os.path.isdir(d) and d not in sys.path:
            site.addsitedir(d)  # append (no stdlib shadowing) + process .pth files


def _detect_benchmark():
    try:
        a0 = sys.argv[0]
    except Exception:
        return None
    if not a0:
        return None
    d = os.path.basename(os.path.dirname(os.path.abspath(a0)))
    return d[3:] if d.startswith("bm_") else None


def _banner(mode):
    # One-line, machine-greppable proof of what this worker actually did. Emitted
    # to stderr when BENCH_CINDERX_BANNER=1, and (because pyperf hides worker
    # stderr on success) also appended to BENCH_CINDERX_BANNER_FILE when set, so
    # the real benchmark workers leave an auditable trail.
    want = os.environ.get("BENCH_CINDERX_BANNER", "") == "1"
    fpath = os.environ.get("BENCH_CINDERX_BANNER_FILE", "")
    if not want and not fpath:
        return
    loaded = 1 if "cinderx" in sys.modules else 0
    enabled = 0
    ver = "?"
    if loaded:
        try:
            import importlib.metadata as _md
            ver = _md.version("cinderx")
        except Exception:
            ver = getattr(sys.modules.get("cinderx"), "__version__", None) or "?"
        try:
            import cinderx.jit as _j
            # NOTE: is_enabled() is True merely from importing cinderx, so it does
            # NOT distinguish nojit from auto. The real signal for "the JIT will
            # actually compile code" is an armed auto-threshold OR a loaded jit
            # list. jit=1 here means auto/jitlist is truly active; nojit -> 0.
            _armed = _j.get_compile_after_n_calls() is not None
            _listed = bool(_j.get_jit_list() or [])
            enabled = 1 if (_armed or _listed) else 0
        except Exception:
            pass
    bench = _detect_benchmark() or (os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "?")
    line = ("cinderx-banner: mode=%s loaded=%d jit=%d version=%s bench=%s pid=%d"
            % (mode, loaded, enabled, ver, bench, os.getpid()))
    if want:
        print(line, file=sys.stderr)
    if fpath:
        try:
            with open(fpath, "a") as fh:  # O_APPEND: concurrent short-line writes are atomic
                fh.write(line + "\n")
        except Exception:
            pass


def _setup():
    # Always run first, even for the "off" configs: this only extends sys.path so
    # that *if* a policy wants cinderx it is importable; it does not enable it.
    _add_cinderx_site()

    if os.environ.get("CINDERX_DISABLE", "") == "1":
        _banner("off")
        return  # (a) no CinderX at all
    try:
        import cinderx  # noqa: F401  (init the runtime even in no-jit mode)
        import cinderx.jit as jit
    except Exception as e:
        print("sitecustomize: cinderx import failed: %r" % (e,), file=sys.stderr)
        _banner("import-failed")
        return

    if os.environ.get("CINDERX_NO_JIT", "") == "1":
        # (b) cinderx imported, but the JIT is left disabled on purpose.
        if os.environ.get("BENCH_JITLIST_VERBOSE") == "1":
            print("sitecustomize: cinderx imported, JIT NOT enabled", file=sys.stderr)
        _banner("nojit")
        return

    jitdir = os.environ.get("BENCH_JITLIST_DIR", "")
    if not jitdir:
        jit.auto()  # (d) baseline
        _banner("auto")
        return

    # (c) per-benchmark JIT list
    name = _detect_benchmark()
    path = os.path.join(jitdir, name + ".jitlist") if name else ""
    if path and os.path.isfile(path):
        try:
            jit.read_jit_list(path)  # NOTE: no auto() on purpose
            if os.environ.get("BENCH_JITLIST_VERBOSE") == "1":
                print("sitecustomize: JIT list %s" % path, file=sys.stderr)
            _banner("jitlist")
            return
        except Exception as e:
            print("sitecustomize: read_jit_list(%s) failed: %r" % (path, e),
                  file=sys.stderr)
    # No list for this benchmark.
    if os.environ.get("BENCH_JITLIST_FALLBACK", "none") == "auto":
        jit.auto()
        _banner("jitlist-fallback-auto")
    else:
        # JIT enabled, no list + no threshold -> compiles nothing (isolates effect)
        _banner("jitlist-none")


_setup()
PYEOF

  # --- report_8way.py : N-way comparison table (used by every suite) ------
  cat > "$HELPERS/report_8way.py" <<'PYEOF'
#!/usr/bin/env python3
"""Aggregate a <name>\t<config>\t<trial>\t<status>\t<seconds> TSV into a markdown
table with one column per configuration plus a geomean-speedup row.

Usage: report_8way.py <tsv> <title> <out.md> <config1,config2,...>
The first config in the list is the baseline; speedup = baseline / config
(>1 = that config is faster than the baseline). Cell = best (min) seconds over
trials (lower = better)."""
import csv, sys, statistics, collections, math

tsv, title, out, config_csv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
configs = [c for c in config_csv.split(",") if c]
baseline = configs[0] if configs else None

g = collections.defaultdict(list)   # (name, config) -> [seconds]
order = []
with open(tsv) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["name"] not in order:
            order.append(r["name"])
        if r.get("status", "OK") == "OK" and r.get("seconds"):
            g[(r["name"], r["config"])].append(float(r["seconds"]))


def best(name, cfg):
    xs = g.get((name, cfg))
    return min(xs) if xs else None


header = "| Benchmark | " + " | ".join(configs) + " |"
sep = "|" + "---|" * (len(configs) + 1)
lines = [f"# {title}\n",
         "Cell = best (min) wall-clock seconds over trials (lower = better). "
         f"Baseline = `{baseline}`; the geomean row shows speedup vs the "
         "baseline (>1× = faster than baseline).\n",
         header, sep]

ratios = {c: [] for c in configs}
for name in order:
    b = best(name, baseline)
    cells = []
    for c in configs:
        v = best(name, c)
        if v is None:
            cells.append("—")
            continue
        cells.append(f"{v:.3f}")
        if b and v and c != baseline:
            ratios[c].append(b / v)
    lines.append("| " + name + " | " + " | ".join(cells) + " |")

geo_cells = []
for c in configs:
    if c == baseline:
        geo_cells.append("1.000×")
    elif ratios[c]:
        geo = math.exp(statistics.fmean(math.log(x) for x in ratios[c]))
        geo_cells.append(f"{geo:.3f}×")
    else:
        geo_cells.append("—")
lines.append("| **geomean speedup vs baseline** | " + " | ".join(geo_cells) + " |")

text = "\n".join(lines) + "\n"
with open(out, "w") as f:
    f.write(text)
print(text)
PYEOF

  # --- cinderx_shim/ : no-op cinderx fallback for Suite A "off" configs ----
  # The built-in benchmark scripts (binary_trees.py, ...) are downloaded source
  # we may not modify, and they do `import cinderx.jit` + `cinderx.jit.auto()`
  # UNCONDITIONALLY at module scope. On an interpreter that has no real cinderx
  # (the plain venv when no dynamic CinderX wheel is installed) that import is a
  # hard ModuleNotFoundError, so the benchmark can't even start.
  #
  # For the "off" configurations (CINDERX_DISABLE=1) we don't WANT the JIT anyway
  # -- "off" is the plain-CPython baseline. This shim provides a minimal cinderx
  # package whose jit.auto() (and friends) are no-ops, so the untouched benchmark
  # imports cleanly and runs as pure CPython. It is placed on PYTHONPATH ONLY when
  # the interpreter cannot import the REAL cinderx AND the config is an "off" one,
  # so it never shadows a real (builtin/static or dynamic-wheel) cinderx.
  local shimdir="$HELPERS/cinderx_shim/cinderx"
  mkdir -p "$shimdir"
  cat > "$shimdir/__init__.py" <<'PYEOF'
# No-op cinderx shim (Suite A "off" configs only). See cinderx-benchmark.sh.
# Present so `import cinderx` / `import cinderx.jit` succeed with the JIT disabled.
__version__ = "0.0.0+shim"
PYEOF
  cat > "$shimdir/jit.py" <<'PYEOF'
# No-op cinderx.jit shim: every entry point is a harmless no-op so the untouched
# benchmark scripts (which call cinderx.jit.auto() unconditionally) run as plain
# CPython. Used ONLY for "off" configs on interpreters lacking a real cinderx.
def auto(*args, **kwargs):
    return None


def is_enabled(*args, **kwargs):
    return False


def disable(*args, **kwargs):
    return None


def read_jit_list(*args, **kwargs):
    return None


def get_jit_list(*args, **kwargs):
    return []


def get_compiled_functions(*args, **kwargs):
    return []


def get_compile_after_n_calls(*args, **kwargs):
    return None
PYEOF

  ok "Helpers written"
}

###############################################################################
# Phase 1: build CPython with PGO + LTO.
###############################################################################
build_cpython() {
  if [ "$DO_BUILD_CPYTHON" -eq 0 ]; then
    warn "Skipping CPython build (--skip-build-cpython)"; return
  fi
  if [ -x "$PY_PREFIX/bin/python3" ]; then
    ok "Plain CPython already built at $PY_PREFIX ($("$PY_PREFIX/bin/python3" -V 2>&1)); skipping"
    return
  fi
  log "Cloning CPython $CPYTHON_TAG from $CPYTHON_REPO"
  if [ ! -d "$SRC_CPYTHON/.git" ]; then
    # CPython release tags (v3.14.5, ...) are ANNOTATED tags: the ref points at a
    # tag OBJECT, not a commit. `git clone --depth 1 --branch <annotated-tag>`
    # does shallow-clone the right tree, but the shallow negotiation lists that
    # tag ref and git prints "warning: refs/tags/<tag> <sha> is not a commit!".
    # Fetch the ref explicitly instead and check out the commit it resolves to:
    # FETCH_HEAD is always dereferenced to a commit, so there is no warning. This
    # form is also transparent for branches and lightweight tags.
    { git init -q "$SRC_CPYTHON" \
        && git -C "$SRC_CPYTHON" remote add origin "$CPYTHON_REPO" \
        && git -C "$SRC_CPYTHON" fetch --depth 1 origin "$CPYTHON_TAG" \
        && git -C "$SRC_CPYTHON" -c advice.detachedHead=false checkout --detach FETCH_HEAD ; } \
      2>&1 | tee "$LOGDIR/cpython_clone.log" || die "CPython clone failed"
  else
    ok "CPython checkout already present at $SRC_CPYTHON"
  fi

  # Build CPython with the SAME compiler we use for the CinderX archives so their
  # libstdc++ versions match (this is what removes the static libstdc++.a need).
  # configure records CC/CXX in the Makefile, so the subsequent make uses them too.
  detect_toolchain
  local cc_args=()
  [ -n "$TOOLCHAIN_CC" ]  && cc_args+=("CC=$TOOLCHAIN_CC")
  [ -n "$TOOLCHAIN_CXX" ] && cc_args+=("CXX=$TOOLCHAIN_CXX")
  # When BOLT is enabled, link with -Wl,--emit-relocs so llvm-bolt can run in
  # relocation mode (required for function reordering / cold splitting). configure
  # records LDFLAGS into the Makefile, so both the PGO `make` here AND the static
  # `make python` relink (build_static_cpython) inherit the flag — the static
  # binary therefore also carries relocations for its own BOLT pass. Any existing
  # $LDFLAGS is preserved.
  local extra_conf=()
  if [ "$DO_BOLT" -eq 1 ]; then
    extra_conf+=("LDFLAGS=-Wl,--emit-relocs${LDFLAGS:+ $LDFLAGS}")
    log "BOLT enabled: linking CPython with -Wl,--emit-relocs (BOLT relocation mode)"
  fi
  if [ "${#cc_args[@]}" -gt 0 ]; then
    log "Configuring CPython with PGO + LTO and ${cc_args[*]}"
  else
    log "Configuring CPython with --enable-optimizations (PGO) + --with-lto"
  fi
  ( cd "$SRC_CPYTHON" \
    && ./configure --prefix="$PY_PREFIX" --enable-optimizations --with-lto \
         "${cc_args[@]}" "${extra_conf[@]}" \
         >"$LOGDIR/cpython_configure.log" 2>&1 ) \
    || die "CPython configure failed (see $LOGDIR/cpython_configure.log)"

  log "Building CPython (make -j$JOBS, includes PGO instrument+train pass — this is slow)"
  ( cd "$SRC_CPYTHON" && make -j"$JOBS" >"$LOGDIR/cpython_make.log" 2>&1 ) \
    || die "CPython build failed (see $LOGDIR/cpython_make.log)"

  log "Installing CPython into $PY_PREFIX (make install)"
  ( cd "$SRC_CPYTHON" && make install >"$LOGDIR/cpython_install.log" 2>&1 ) \
    || die "CPython install failed (see $LOGDIR/cpython_install.log)"

  [ -x "$PY_PREFIX/bin/python3" ] || die "python3 missing after install"
  ok "Plain CPython built: $("$PY_PREFIX/bin/python3" -V 2>&1)"
}

###############################################################################
# Phase 1b: build the SECOND, static CPython (builtin _cinderx) into its own
# prefix. Always runs (the static interpreter is no longer optional).
#
# CinderX's cmake already bundles all module logic into libcinderx-lib.a + a set
# of static sub-archives; the only thing missing for a builtin module is the
# PyInit__cinderx entry point (normally compiled straight into _cinderx.so). We
# build the .a's from source, compile a tiny PyInit__cinderx wrapper out-of-band
# (makesetup's .cpp rule injects C-only -std=c11 that a C++ compiler rejects),
# declare _cinderx in Modules/Setup.local with all archives wrapped in
# --start-group/--end-group (they have circular refs), and relink with
# `make python`. The plain prefix is then copied to the static prefix and its
# interpreter binary swapped for the relinked one (CPython is a static-libpython
# build, so copying the binary suffices). The pure-Python cinderx package is
# added to the static venv separately (install_cinderx_pythonlib).
###############################################################################

# Map a C++ compiler name/path to its sibling C compiler (g++ -> gcc, clang++ ->
# clang). Preserves any directory prefix so custom toolchain installs work too.
cc_for_cxx() {
  case "$1" in
    *clang++*) echo "${1%clang++}clang" ;;
    *g++*)     echo "${1%g++}gcc" ;;
    *c++*)     echo "${1%c++}cc" ;;
    *)         echo cc ;;
  esac
}

# Return 0 if compiler "$1" can actually compile a small C++20 program. We probe
# with a real compile (concepts + <version>) instead of parsing version numbers,
# so any toolchain that genuinely supports C++20 is accepted regardless of vendor
# or version string.
cxx_supports_cxx20() {
  local cxx="$1" tmp rc
  command -v "$cxx" >/dev/null 2>&1 || [ -x "$cxx" ] || return 1
  tmp="$(mktemp 2>/dev/null)" || return 1
  # GCC accepts -std=c++20 and compiles concepts as far back as gcc 10, but its
  # C++20 standard-library/feature support is not complete enough for CinderX
  # until gcc 13. So the compile probe alone (concepts + <version>) would wrongly
  # accept gcc 11/12. Additionally gate GCC to >= 13 via __GNUC__; Clang (which
  # advertises a low __GNUC__ for GCC compatibility but defines __clang__) and
  # other vendors are left to the genuine C++20 feature probe below.
  printf '%s\n' \
    '#include <version>' \
    '#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 13)' \
    '#error "GCC < 13 does not support C++20 well enough for CinderX (need gcc 13+)"' \
    '#endif' \
    'template <class T> concept Addable = requires(T a, T b) { a + b; };' \
    'template <Addable T> T add(T a, T b) { return a + b; }' \
    'int main() { return add(0, 0); }' \
    | "$cxx" -std=c++20 -x c++ -c -o "$tmp" - >/dev/null 2>&1
  rc=$?
  rm -f "$tmp"
  return $rc
}

# Detect ONE C++20 compiler (and its sibling C compiler) to build BOTH CPython
# and the CinderX archives with. Using the same toolchain for both keeps their
# libstdc++ versions in lockstep, which is what lets us link the C++ runtime
# dynamically (-lstdc++) instead of bundling a static libstdc++.a. Populates
# TOOLCHAIN_CC / TOOLCHAIN_CXX, plus TOOLCHAIN_LIBSTDCXX (a static libstdc++.a
# kept only as a fallback for the static link, used if dynamic linking fails).
#
# Everything here is generic: no hardcoded toolchain paths, so it works on any
# Linux (Ubuntu/Debian system gcc, Fedora/RHEL gcc-toolset on PATH, clang, etc.).
# Honours CINDERX_CXX / CINDERX_CC / CINDERX_LIBSTDCXX_A overrides in all modes.
# Validate an explicitly-provided compiler (via --cc/--cxx or CINDERX_CC/CINDERX_CXX)
# exists and is executable. Accepts a full path (e.g. /opt/gcc-15/bin/gcc, checked
# with -x) or a bare command name resolved on PATH (via command -v). $3 is a label
# describing where the value came from, for a clear error message.
validate_compiler() {
  local what="$1" comp="$2" origin="$3"
  case "$comp" in
    */*) [ -x "$comp" ] || die "$what from $origin does not exist or is not executable: $comp" ;;
    *)   command -v "$comp" >/dev/null 2>&1 || die "$what from $origin not found on PATH: $comp" ;;
  esac
}

detect_toolchain() {
  # Idempotent: detection is cheap but several phases call this.
  [ -n "$TOOLCHAIN_CXX" ] && return 0

  # Validate any explicitly-provided compiler up front (CLI flags or env vars).
  # These override auto-detection, so a bad path should fail loudly rather than
  # silently fall through to a probed compiler.
  [ -n "$CINDERX_CC" ]  && validate_compiler "C compiler"   "$CINDERX_CC"  "$CC_ORIGIN"
  [ -n "$CINDERX_CXX" ] && validate_compiler "C++ compiler" "$CINDERX_CXX" "$CXX_ORIGIN"

  if [ -n "$CINDERX_CXX" ]; then
    # Explicit override: trust the caller's choice of C++ compiler.
    TOOLCHAIN_CXX="$CINDERX_CXX"
    TOOLCHAIN_CC="${CINDERX_CC:-$(cc_for_cxx "$CINDERX_CXX")}"
  else
    # Probe the usual C++ drivers and pick the first that supports C++20. To use
    # a gcc-toolset (or any other) compiler, put it on PATH or set CINDERX_CXX.
    local cxx
    for cxx in g++ clang++ c++; do
      if cxx_supports_cxx20 "$cxx"; then
        TOOLCHAIN_CXX="$(command -v "$cxx")"
        TOOLCHAIN_CC="${CINDERX_CC:-$(command -v "$(cc_for_cxx "$cxx")" 2>/dev/null)}"
        break
      fi
    done
  fi

  if [ -n "$CINDERX_LIBSTDCXX_A" ]; then
    TOOLCHAIN_LIBSTDCXX="$CINDERX_LIBSTDCXX_A"
  elif [ -n "$TOOLCHAIN_CXX" ]; then
    # Ask the chosen compiler where its own static libstdc++.a lives. This is
    # portable across distros and toolchains. If the static lib is not installed
    # the compiler just echoes back the bare name, so confirm it is a real file.
    local cand
    cand="$("$TOOLCHAIN_CXX" -print-file-name=libstdc++.a 2>/dev/null)"
    [ -n "$cand" ] && [ -f "$cand" ] && TOOLCHAIN_LIBSTDCXX="$cand"
  fi
  return 0
}

build_static_cpython() {
  if [ "$DO_BUILD_CPYTHON" -eq 0 ]; then
    warn "Skipping static CPython build (--skip-build-cpython)"; return
  fi
  log "Static CPython: link _cinderx into a second interpreter as a builtin module"

  # Idempotent: if the static prefix already has the builtin, nothing to do.
  if [ -x "$PY_PREFIX_STATIC/bin/python3" ] && \
     "$PY_PREFIX_STATIC/bin/python3" -c "import sys; sys.exit(0 if '_cinderx' in sys.builtin_module_names else 1)" 2>/dev/null; then
    ok "_cinderx already statically linked into $PY_PREFIX_STATIC/bin/python3; skipping"
    return
  fi

  [ -x "$PY_PREFIX/bin/python3" ] || die "No plain CPython at $PY_PREFIX; build it first (drop --skip-build-cpython)"
  [ -d "$SRC_CPYTHON" ] || die "No CPython build tree at $SRC_CPYTHON; the static build needs the source/build tree to relink"
  [ -f "$SRC_CPYTHON/Makefile" ] || die "No Makefile in $SRC_CPYTHON; CPython was not configured/built there"

  detect_toolchain
  [ -n "$TOOLCHAIN_CXX" ] || die "no C++20 compiler available for the static build (need gcc 13+ or clang)"

  # CinderX source checkout (reuse the source-build clone path / settings).
  if [ ! -d "$SRC_CINDERX/.git" ] && [ ! -f "$SRC_CINDERX/CMakeLists.txt" ]; then
    log "Cloning CinderX $CINDERX_TAG from $CINDERX_REPO"
    git clone --depth 1 --branch "$CINDERX_TAG" "$CINDERX_REPO" "$SRC_CINDERX" \
      2>&1 | tee "$LOGDIR/cinderx_clone.log" || die "CinderX clone failed"
  else
    ok "CinderX checkout present at $SRC_CINDERX"
  fi
  [ -f "$SRC_CINDERX/CMakeLists.txt" ] || die "CinderX CMakeLists.txt not found in $SRC_CINDERX"

  # Match the build to the interpreter we link into.
  local pyver ft
  pyver="$("$PY_PREFIX/bin/python3" -c 'import sys;print("%d.%d"%sys.version_info[:2])')" \
    || die "could not determine CPython version"
  ft="$("$PY_PREFIX/bin/python3" -c 'import sysconfig;print(1 if sysconfig.get_config_var("Py_GIL_DISABLED") else 0)')"

  # Build the CinderX static archives with cmake (option matrix mirrors CinderX's
  # setup.py for a stock 3.14+ non-meta interpreter; cinderx-lib transitively
  # builds every sub-archive + vendored asmjit/fmt/capstone).
  log "Building CinderX static archives via cmake (py=$pyver free_threading=$ft, CXX=$TOOLCHAIN_CXX) -> $STATIC_BUILD_DIR"
  mkdir -p "$STATIC_BUILD_DIR"
  cmake -G Ninja -B "$STATIC_BUILD_DIR" "$SRC_CINDERX" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_C_COMPILER="$TOOLCHAIN_CC" \
      -DCMAKE_CXX_COMPILER="$TOOLCHAIN_CXX" \
      -DPY_VERSION="$pyver" \
      -DPython_ROOT_DIR="$PY_PREFIX" \
      -DENABLE_FREE_THREADING="$ft" \
      -DMETA_PYTHON=0 \
      -DENABLE_ADAPTIVE_STATIC_PYTHON=0 \
      -DENABLE_DISASSEMBLER=1 \
      -DENABLE_ELF_READER=1 \
      -DENABLE_EVAL_HOOK=0 \
      -DENABLE_FUNC_EVENT_MODIFY_QUALNAME=0 \
      -DENABLE_GENERATOR_AWAITER=0 \
      -DENABLE_INTERPRETER_LOOP=1 \
      -DENABLE_LAZY_IMPORTS=0 \
      -DENABLE_LIGHTWEIGHT_FRAMES=0 \
      -DENABLE_PARALLEL_GC=0 \
      -DENABLE_PEP523_HOOK=1 \
      -DENABLE_PERF_TRAMPOLINE=0 \
      -DENABLE_SYMBOLIZER=1 \
      -DENABLE_USDT=1 \
      -DENABLE_ZLIB=1 \
      >"$LOGDIR/static_cmake_configure.log" 2>&1 \
    || die "CinderX cmake configure failed (see $LOGDIR/static_cmake_configure.log)"
  cmake --build "$STATIC_BUILD_DIR" --target cinderx-lib -j "$JOBS" \
      >"$LOGDIR/static_cmake_build.log" 2>&1 \
    || die "CinderX cmake build failed (see $LOGDIR/static_cmake_build.log)"

  # Collect every produced static archive (robust against layout changes).
  local archives=()
  while IFS= read -r a; do archives+=("$a"); done \
    < <(find "$STATIC_BUILD_DIR" -name '*.a' | sort)
  [ "${#archives[@]}" -gt 0 ] || die "no .a archives produced under $STATIC_BUILD_DIR"
  printf '%s\n' "${archives[@]}" | grep -q 'libcinderx-lib\.a$' \
    || die "libcinderx-lib.a not found among produced archives"
  ok "Built ${#archives[@]} CinderX static archives"

  # Compile the PyInit__cinderx wrapper out-of-band. It MUST be C++ — the bridged
  # _cinderx_lib_init() has C++ (mangled) linkage in the archive. Compiled here
  # (not via makesetup) because makesetup's .cpp rule injects C-only -std=c11.
  local wrapper_cpp="$STATIC_BUILD_DIR/_cinderx_static.cpp"
  local wrapper_o="$STATIC_BUILD_DIR/_cinderx_static.o"
  cat > "$wrapper_cpp" <<'CPPEOF'
// Static-link entry point for the builtin _cinderx module (generated by
// cinderx-benchmark.sh).
//
// All CinderX module logic lives in libcinderx-lib.a, but PyInit__cinderx
// normally lives in cinderx/_cinderx.cpp which is only compiled into _cinderx.so.
// For a builtin (static) module we supply that entry point here; CPython's
// generated inittab (Modules/config.c) calls it.
//
// _cinderx_lib_init() has C++ linkage in the archive, so this file MUST be
// compiled as C++. PyInit__cinderx itself is extern "C" (PyMODINIT_FUNC), so the
// C-compiled config.c resolves it by its unmangled name.
#include <Python.h>

// Defined in libcinderx-lib.a.
PyObject* _cinderx_lib_init();

PyMODINIT_FUNC PyInit__cinderx(void) {
  return _cinderx_lib_init();
}
CPPEOF
  log "Compiling PyInit__cinderx wrapper ($TOOLCHAIN_CXX, out-of-band)"
  "$TOOLCHAIN_CXX" -std=c++20 -fPIC -O2 \
      -I"$PY_PREFIX/include/python$pyver" \
      -c "$wrapper_cpp" -o "$wrapper_o" \
      >"$LOGDIR/static_wrapper_compile.log" 2>&1 \
    || die "wrapper compile failed (see $LOGDIR/static_wrapper_compile.log)"

  # Declare _cinderx as a builtin static module in Modules/Setup.local. The
  # archives are wrapped in --start-group/--end-group to resolve their circular
  # references; the static libstdc++.a (if found) goes inside the group so the
  # C++ runtime is linked in statically. -lz/-lm satisfy CinderX's deps.
  # Link the builtin _cinderx. Because CPython and the CinderX archives are now
  # built with the SAME compiler (detect_toolchain), their libstdc++ versions
  # match, so we link the C++ runtime DYNAMICALLY (-lstdc++) by default. Only if
  # that link fails do we fall back to bundling the static libstdc++.a inside the
  # archive group. The archives are wrapped in --start-group/--end-group for their
  # circular references; -lz/-lm satisfy CinderX's deps.
  local link_modes=(dynamic)
  [ -n "$TOOLCHAIN_LIBSTDCXX" ] && link_modes+=(static)   # fallback only

  local linked=0 mode a
  for mode in "${link_modes[@]}"; do
    log "Writing $SRC_CPYTHON/Modules/Setup.local (libstdc++: $mode)"
    {
      echo "# Auto-generated by cinderx-benchmark.sh — statically link CinderX."
      echo "# Wrapper .o (PyInit__cinderx) compiled out-of-band; archives from CinderX cmake."
      echo "# Archives wrapped in --start-group/--end-group for their circular references."
      printf '_cinderx %s -Wl,--start-group' "$wrapper_o"
      for a in "${archives[@]}"; do printf ' %s' "$a"; done
      # static mode bundles libstdc++.a inside the group; dynamic links -lstdc++.
      [ "$mode" = static ] && printf ' %s' "$TOOLCHAIN_LIBSTDCXX"
      printf ' -Wl,--end-group'
      [ "$mode" = dynamic ] && printf ' -lstdc++'
      printf ' -lz -lm\n'
    } > "$SRC_CPYTHON/Modules/Setup.local"

    # Relink. Use `make python` (NOT plain `make`, which would redo the full PGO
    # instrument+train pass). Editing Setup.local makes the Makefile regenerate
    # itself on the first invocation; run again so the new config.c/_cinderx links.
    log "Relinking CPython with the builtin _cinderx (make python -j$JOBS, libstdc++: $mode)"
    ( cd "$SRC_CPYTHON" && make python -j"$JOBS" ) \
        >"$LOGDIR/static_make_python.log" 2>&1 || true
    if ! "$SRC_CPYTHON/python" -c "import sys; sys.exit(0 if '_cinderx' in sys.builtin_module_names else 1)" 2>/dev/null; then
      log "  (re-running make python after Makefile regeneration)"
      ( cd "$SRC_CPYTHON" && make python -j"$JOBS" ) \
          >>"$LOGDIR/static_make_python.log" 2>&1 || true
    fi
    if "$SRC_CPYTHON/python" -c "import sys; sys.exit(0 if '_cinderx' in sys.builtin_module_names else 1)" 2>/dev/null; then
      linked=1
      ok "Linked builtin _cinderx (libstdc++: $mode)"
      break
    fi
    warn "Relink with libstdc++:$mode failed (see $LOGDIR/static_make_python.log)"
  done
  [ "$linked" -eq 1 ] \
    || die "could not link builtin _cinderx with dynamic or static libstdc++ (see $LOGDIR/static_make_python.log)"

  # Materialise the static interpreter in its OWN prefix: copy the entire plain
  # prefix (stdlib, headers, pip, ...) then swap in the freshly relinked binary.
  # CPython here is a static-libpython build (no libpython*.so), so copying the
  # binary over the copied tree suffices and avoids `make install` retriggering
  # the PGO build. getpath finds the static prefix from the executable location.
  log "Installing the static interpreter into $PY_PREFIX_STATIC (copy plain prefix + swap binary)"
  rm -rf "$PY_PREFIX_STATIC"
  cp -a "$PY_PREFIX" "$PY_PREFIX_STATIC" || die "failed to copy $PY_PREFIX -> $PY_PREFIX_STATIC"
  local target
  target="$(readlink -f "$PY_PREFIX_STATIC/bin/python$pyver")"
  [ -n "$target" ] || target="$PY_PREFIX_STATIC/bin/python$pyver"
  cp -f "$SRC_CPYTHON/python" "$target" || die "failed to copy static python into $target"
  "$PY_PREFIX_STATIC/bin/python3" -c "import sys; sys.exit(0 if '_cinderx' in sys.builtin_module_names else 1)" 2>/dev/null \
    || die "_cinderx not builtin in $PY_PREFIX_STATIC/bin/python3 after install"
  ok "Static CPython built: _cinderx is builtin in $("$PY_PREFIX_STATIC/bin/python3" -V 2>&1) at $PY_PREFIX_STATIC"
}

# Static venv: make the pure-Python cinderx package (PythonLib) importable via a
# .pth file. The builtin _cinderx (in the static interpreter) provides the native
# side; the PythonLib provides `import cinderx` / `cinderx.jit`.
install_cinderx_pythonlib() {
  [ -n "$VPY_STATIC" ] || die "static venv not ready"
  # Idempotent: if the static venv can already import cinderx, nothing to do.
  if "$VPY_STATIC" -c 'import cinderx.jit' >/dev/null 2>&1; then
    ok "Static venv can already import cinderx; skipping PythonLib .pth"
    return
  fi
  local lib="$SRC_CINDERX/cinderx/PythonLib"
  [ -d "$lib/cinderx" ] || die "CinderX PythonLib not found at $lib (expected after the source checkout)"
  local sp
  sp="$("$VPY_STATIC" -c 'import site; print(site.getsitepackages()[0])')" \
    || die "could not locate site-packages"
  echo "$lib" > "$sp/cinderx_pythonlib.pth"
  # Verify the builtin native module + pure-Python package import together.
  "$VPY_STATIC" -c "import sys; assert '_cinderx' in sys.builtin_module_names, 'no builtin _cinderx'; import cinderx.jit as j; print('static cinderx OK (builtin _cinderx + PythonLib)')" \
      >>"$LOGDIR/cinderx_verify.log" 2>&1 \
    || die "static cinderx import failed (builtin _cinderx + PythonLib on $lib); see $LOGDIR/cinderx_verify.log"
  ok "Pure-Python cinderx package on static venv path via $sp/cinderx_pythonlib.pth"
}

###############################################################################
# Phase 2: create the TWO benchmark venvs (plain + static) and install deps.
###############################################################################

# Create one benchmark venv from $1=interpreter-prefix at $2=venv-path and
# install pyperformance + the fastmark deps + the env-driven sitecustomize.
# Echoes nothing; callers read the venv python from "$2/bin/python".
create_one_venv() {
  local prefix="$1" venv="$2" label="$3"
  [ -x "$prefix/bin/python3" ] || die "No CPython at $prefix; run without --skip-build-cpython"
  log "Creating $label benchmark venv at $venv"
  "$prefix/bin/python3" -m venv "$venv" || die "$label venv creation failed"
  local vpy="$venv/bin/python"
  "$vpy" -m pip install --upgrade pip setuptools wheel \
    >"$LOGDIR/pip_bootstrap_${label}.log" 2>&1 || die "$label pip bootstrap failed"

  log "Installing pyperformance + fastmark benchmark dependencies ($label)"
  # Mirrors cinderx/benchmarks/requirements-fastmark.txt.
  "$vpy" -m pip install \
      "pyperformance==1.14.0" \
      coverage docutils dulwich genshi html5lib mako pyaes \
      "sqlalchemy<2.0" sqlglot sympy tomli websockets \
      >"$LOGDIR/pip_bench_deps_${label}.log" 2>&1 \
    || die "$label benchmark dependency install failed (see $LOGDIR/pip_bench_deps_${label}.log)"
  ok "Benchmark dependencies installed ($label)"

  install_sitecustomize "$vpy"
}

make_venvs() {
  # Plain venv (dynamic CinderX wheel goes in later via install_cinderx).
  if [ "$DO_VENV" -eq 0 ] && [ -x "$VENV/bin/python" ]; then
    VPY="$VENV/bin/python"; ok "Reusing existing plain venv $VENV"
  else
    create_one_venv "$PY_PREFIX" "$VENV" "plain"
    VPY="$VENV/bin/python"
  fi

  # Static venv (PythonLib .pth goes in later via install_cinderx_pythonlib).
  if [ "$DO_VENV" -eq 0 ] && [ -x "$VENV_STATIC/bin/python" ]; then
    VPY_STATIC="$VENV_STATIC/bin/python"; ok "Reusing existing static venv $VENV_STATIC"
  else
    create_one_venv "$PY_PREFIX_STATIC" "$VENV_STATIC" "static"
    VPY_STATIC="$VENV_STATIC/bin/python"
  fi
}

# Place the env-driven sitecustomize.py into a venv's site-packages.
# $1 = venv python interpreter.
install_sitecustomize() {
  local vpy="$1" sp
  sp="$("$vpy" -c 'import site; print(site.getsitepackages()[0])')" \
    || die "could not locate site-packages"
  cp "$HELPERS/sitecustomize.py" "$sp/sitecustomize.py"
  ok "Installed env-driven sitecustomize.py into $sp"
}

###############################################################################
# Phase 3: install the dynamic CinderX into the PLAIN venv (PyPI wheel by
# default, or built from a git checkout with --cinderx-source). The static venv
# instead gets the pure-Python PythonLib via install_cinderx_pythonlib (its
# native side is the builtin _cinderx in the static interpreter).
###############################################################################
install_cinderx() {
  if [ "$DO_INSTALL_CINDERX" -eq 0 ]; then
    warn "Skipping dynamic CinderX install (--skip-cinderx)"; return
  fi
  [ -n "$VPY" ] || die "plain venv not ready"

  # Idempotency probe MUST use the plain venv python ($VPY) and MUST test
  # 'import cinderx.jit', not the bare 'import cinderx'. The top-level package is
  # pure Python and can import even when the native runtime is missing; only
  # 'cinderx.jit' (which needs the _cinderx extension) proves CinderX is actually
  # usable. This matches sitecustomize, the post-install verify, and the JIT-list
  # generator -- all of which require 'import cinderx.jit'. Using the weaker probe
  # here caused a false "already importable; skipping install" while the workers'
  # sitecustomize reported "cinderx import failed: ModuleNotFoundError".
  if "$VPY" -c 'import cinderx.jit' >/dev/null 2>&1; then
    ok "Dynamic CinderX already importable in plain venv; skipping install"
    return
  fi

  if [ "$CINDERX_SOURCE" = "1" ]; then
    log "Building CinderX from source ($CINDERX_REPO@$CINDERX_TAG)"
    if [ ! -d "$SRC_CINDERX/.git" ]; then
      git clone --depth 1 --branch "$CINDERX_TAG" "$CINDERX_REPO" "$SRC_CINDERX" \
        2>&1 | tee "$LOGDIR/cinderx_clone.log" || die "CinderX clone failed"
    fi
    "$VPY" -m pip install setuptools \
      >"$LOGDIR/cinderx_setuptools.log" 2>&1 || die "setuptools install failed"
    ( cd "$SRC_CINDERX" \
      && "$VPY" -m pip install -e . --no-build-isolation --reinstall \
           >"$LOGDIR/cinderx_build.log" 2>&1 ) \
      || die "CinderX source build failed (see $LOGDIR/cinderx_build.log)"
  else
    local spec="cinderx"
    [ -n "$CINDERX_VERSION" ] && spec="cinderx==$CINDERX_VERSION"
    log "Installing CinderX from PyPI ($spec)"
    "$VPY" -m pip install "$spec" \
      >"$LOGDIR/cinderx_pip.log" 2>&1 \
      || die "CinderX PyPI install failed (see $LOGDIR/cinderx_pip.log). \
Try --cinderx-source to build from git, or check your CPython is 3.14 with a compatible wheel."
  fi

  # Verify both ON and OFF states work.
  "$VPY" -c 'import cinderx.jit as j; print("cinderx import OK")' \
    >>"$LOGDIR/cinderx_verify.log" 2>&1 || die "cinderx import failed after install"
  local ver
  ver="$("$VPY" -m pip show cinderx 2>/dev/null | awk '/^Version:/{print $2}')"
  ok "CinderX installed (version ${ver:-unknown})"
}

# Ensure a CinderX *source* checkout exists at $SRC_CINDERX. The static build and
# the static venv's PythonLib need it, and benchmark suites A/B/C run scripts from
# the source tree's cinderx/benchmarks/ directory, which is NOT included in the
# PyPI wheel. Idempotent (build_static_cpython / --cinderx-source may already have
# cloned here; this just fills the gap otherwise).
ensure_cinderx_source() {
  if [ -d "$SRC_CINDERX/.git" ] || [ -f "$SRC_CINDERX/CMakeLists.txt" ]; then
    return 0
  fi
  log "Cloning CinderX source $CINDERX_TAG from $CINDERX_REPO (for benchmark suites A/B/C)"
  if git clone --depth 1 --branch "$CINDERX_TAG" "$CINDERX_REPO" "$SRC_CINDERX" \
       2>&1 | tee "$LOGDIR/cinderx_clone.log"; then
    ok "CinderX source checkout at $SRC_CINDERX"
    return 0
  fi
  warn "CinderX source clone failed (see $LOGDIR/cinderx_clone.log); suites A/B/C will be skipped"
  return 1
}

# Locate the cinderx benchmarks directory. The suites live in the source tree at
# cinderx/benchmarks/ and are NOT shipped in the PyPI wheel, so prefer the source
# checkout; fall back to an installed package that happens to ship them.
find_bench_dir() {
  if [ -d "$SRC_CINDERX/cinderx/benchmarks" ]; then
    printf '%s\n' "$SRC_CINDERX/cinderx/benchmarks"
    return 0
  fi
  "$VPY" - <<'PYEOF' 2>/dev/null
import os, importlib.util
spec = importlib.util.find_spec("cinderx")
if not spec or not spec.submodule_search_locations:
    raise SystemExit(1)
base = list(spec.submodule_search_locations)[0]
cand = os.path.join(base, "benchmarks")
print(cand if os.path.isdir(cand) else "")
PYEOF
}

###############################################################################
# Phase 4: generate per-benchmark JIT lists.
###############################################################################
gen_jitlists() {
  if [ "$DO_JITLISTS" -eq 0 ]; then
    warn "Skipping JIT-list generation (--skip-jitlists)"; return
  fi
  mkdir -p "$JITLIST_DIR"
  if [ "$REGEN_JITLISTS" -eq 0 ] && ls "$JITLIST_DIR"/*.jitlist >/dev/null 2>&1; then
    ok "JIT lists already present in $JITLIST_DIR ($(ls "$JITLIST_DIR"/*.jitlist | wc -l) lists); skipping (use --regen-jitlists to rebuild)"
    return
  fi
  # Pick an interpreter that can actually import cinderx to drive generation.
  # JIT lists are just module:qualname lines (interpreter-agnostic), so either
  # venv can produce them and the result is shared by all eight configs. Prefer
  # the plain venv (matches the dynamic configs); fall back to the static venv,
  # whose builtin _cinderx is always present. Without this, a plain venv that
  # lacks the dynamic CinderX wheel (e.g. a run with --skip-cinderx, or no PyPI
  # wheel for this CPython) makes gen_jitlist.py raise ModuleNotFoundError for
  # EVERY benchmark and silently emit empty lists (the "funcs=0" symptom).
  local genpy="" genlabel=""
  if [ -n "$VPY" ] && "$VPY" -c 'import cinderx.jit' >/dev/null 2>&1; then
    genpy="$VPY"; genlabel="plain venv"
  elif [ -n "$VPY_STATIC" ] && "$VPY_STATIC" -c 'import cinderx.jit' >/dev/null 2>&1; then
    genpy="$VPY_STATIC"; genlabel="static venv"
    warn "Plain venv cannot 'import cinderx'; generating JIT lists with the static venv instead."
    warn "The dynamic configs (dyn-nojit/dyn-auto/dyn-jitlist) will FAIL for the same reason —"
    warn "install the dynamic CinderX into $VENV (re-run without --skip-cinderx; add --cinderx-source if there is no PyPI wheel for this CPython)."
  else
    die "Neither venv can 'import cinderx.jit'; cannot generate JIT lists. Install CinderX first (dynamic wheel into the plain venv and/or the builtin in the static interpreter)."
  fi
  ok "Generating JIT lists with the $genlabel"

  local pp
  pp="$("$genpy" -c 'import pyperformance,os;print(os.path.dirname(pyperformance.__file__))')" \
    || die "pyperformance not importable in the $genlabel; cannot generate JIT lists"
  local D="$pp/data-files/benchmarks"
  log "Generating JIT lists (threshold=$JIT_THRESHOLD, budget=${JIT_BUDGET}s) -> $JITLIST_DIR"

  # Expand the benchmark selection. If it contains the special token "all"
  # (optionally with negative excludes, e.g. "all,-telco,-unpack_sequence"),
  # resolve it through pyperformance itself so we generate JIT lists for exactly
  # the set pyperformance would run: "all" = every benchmark declared in the
  # manifest, minus any -excludes, and filtered to those runnable on this
  # interpreter. `pyperformance list -b <sel>` prints one "- <name>" per line.
  # A plain list of benchmark names is used verbatim (no behaviour change).
  local benches="$PYPERF_BENCHES"
  if printf '%s' ",$PYPERF_BENCHES," | grep -qiE ',[[:space:]]*all[[:space:]]*,'; then
    log "Benchmark list contains 'all'; expanding via pyperformance list -b '$PYPERF_BENCHES'"
    benches="$("$genpy" -m pyperformance list -b "$PYPERF_BENCHES" 2>/dev/null \
                 | sed -n 's/^- //p' | tr '\n' ',')"
    benches="${benches%,}"
    [ -n "$benches" ] || die "could not expand 'all' via 'pyperformance list -b $PYPERF_BENCHES' (is pyperformance installed in the $genlabel?)"
    ok "Expanded 'all' to $(printf '%s' "$benches" | tr ',' '\n' | grep -c .) benchmarks"
  fi

  # Track outcomes so a systemic failure (e.g. cinderx not importable) is loud
  # instead of silently leaving a directory full of empty lists.
  local made=0 failed=0
  local IFS=','
  for b in $benches; do
    local bm="$D/bm_$b/run_benchmark.py"
    if [ ! -f "$bm" ]; then warn "no pyperformance benchmark bm_$b; skipping list"; continue; fi
    local jl="$JITLIST_DIR/$b.jitlist"
    local rc=0
    JIT_THRESHOLD="$JIT_THRESHOLD" "$genpy" "$HELPERS/gen_jitlist.py" "$bm" \
        --budget "$JIT_BUDGET" >"$LOGDIR/gjl_$b.out" 2>"$LOGDIR/gjl_$b.err" || rc=$?
    # Count only real entries (non-comment, non-blank). grep -c already prints 0
    # and exits 1 on no match, so the earlier "|| echo 0" doubled the count.
    local n
    n="$(grep -vcE '^[[:space:]]*(#|$)' "$LOGDIR/gjl_$b.out" 2>/dev/null)"; n="${n:-0}"
    {
      echo "# CinderX JIT list for pyperformance bm_$b"
      echo "# gen_jitlist.py threshold=$JIT_THRESHOLD budget=${JIT_BUDGET}s (via $genlabel)"
      echo "# $(tail -1 "$LOGDIR/gjl_$b.err" 2>/dev/null)"
      echo "# Format: module:qualname (one hot function per line)"
      cat "$LOGDIR/gjl_$b.out"
    } > "$jl"
    if [ "$rc" -ne 0 ]; then
      warn "$(printf '%-16s FAILED (rc=%s; see %s)' "$b" "$rc" "$LOGDIR/gjl_$b.err")"
      failed=$((failed + 1))
    else
      printf '    %-16s funcs=%s\n' "$b" "$n"
      made=$((made + 1))
    fi
  done

  if [ "$made" -eq 0 ]; then
    die "JIT-list generation produced no usable lists ($failed failed); see $LOGDIR/gjl_*.err"
  fi
  [ "$failed" -eq 0 ] || warn "$failed JIT list(s) failed to generate; see $LOGDIR/gjl_*.err"
  ok "JIT lists generated in $JITLIST_DIR ($made ok, $failed failed)"
}

###############################################################################
# Phase 4b: BOLT post-link optimization of the interpreter binaries (opt-in).
#
# BOLT (LLVM Binary Optimization and Layout Tool) rewrites an already-linked
# binary using profile data: it reorders basic blocks (ext-tsp), reorders whole
# functions (cdsort) and splits cold code out of the hot path — exactly the code
# placement that the foobar7/8 analysis showed accounts for ±10% of the
# plain-vs-static gap. We optimize BOTH interpreters in place; because the venvs
# reference the install-prefix python via symlink, swapping the binary makes every
# subsequent benchmark run use the BOLT-optimized interpreter automatically.
#
# Profiling method: INSTRUMENTATION (llvm-bolt -instrument), not perf/perf2bolt.
# Rationale: (1) it needs no perf_event_paranoid relaxation or root; (2) the
# function-reordering/splitting we want requires BOLT's relocation mode either
# way, which we already provide via -Wl,--emit-relocs at link time (see
# build_cpython) — so perf2bolt would offer no advantage here. Each binary gets
# its OWN profile (their hot paths differ). We profile with CINDERX_DISABLE=1 so
# both binaries get a clean, comparable interpreter-core profile: JIT-compiled
# code is generated at runtime and lives outside the ELF, so BOLT cannot touch it
# regardless — the win is in the interpreter/runtime C code layout.
###############################################################################

# readelf shim: prefer binutils readelf, fall back to llvm-readelf. Returns 127
# if neither exists (callers treat that as "cannot verify").
bolt_readelf() {
  if command -v readelf >/dev/null 2>&1; then readelf "$@"
  elif command -v llvm-readelf >/dev/null 2>&1; then llvm-readelf "$@"
  else return 127; fi
}

# BOLT-optimize one interpreter binary IN PLACE.
#   $1 = install prefix (…/python-install or …/python-install-static)
#   $2 = a venv python that has pyperformance installed (for the profile workload)
#   $3 = label ("plain" / "static")
# Best-effort: any failure leaves the ORIGINAL binary untouched (a .prebolt backup
# is kept once we start swapping) and returns without aborting the pipeline.
bolt_one() {
  local prefix="$1" venv="$2" label="$3"
  local py="$prefix/bin/python3"
  [ -x "$py" ] || { warn "BOLT[$label]: no interpreter at $prefix; skipping"; return 0; }

  local pyver bin
  pyver="$("$py" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)" \
    || { warn "BOLT[$label]: could not determine python version; skipping"; return 0; }
  bin="$(readlink -f "$prefix/bin/python$pyver" 2>/dev/null)"
  [ -n "$bin" ] && [ -f "$bin" ] || { warn "BOLT[$label]: real binary python$pyver not found; skipping"; return 0; }

  # Idempotent: a marker next to the binary records a completed BOLT pass.
  if [ -f "$bin.bolt-done" ]; then
    ok "BOLT[$label]: $bin already optimized; skipping"
    return 0
  fi

  # Relocation mode (function reordering) needs -Wl,--emit-relocs -> a .rela.text
  # section survives in the binary. Verify when we can; if the binary was reused
  # from a non-BOLT build it will be missing and we skip gracefully.
  if bolt_readelf -S "$bin" >/dev/null 2>&1; then
    if ! bolt_readelf -S "$bin" 2>/dev/null | grep -qE '\.rela?\.text'; then
      warn "BOLT[$label]: $bin has no .rela.text (linked without -Wl,--emit-relocs)."
      warn "  It was likely built without --bolt. Remove $prefix and re-run with --bolt to rebuild with relocations; BOLT skipped for now."
      return 0
    fi
  else
    warn "BOLT[$label]: cannot verify relocations (no readelf); attempting BOLT anyway."
  fi

  # Locate the pyperformance benchmark scripts + this venv's site-packages so the
  # (standalone) instrumented binary can import pyperf and the benchmark modules.
  local pp bench_d sp
  pp="$("$venv" -c 'import pyperformance,os;print(os.path.dirname(pyperformance.__file__))' 2>/dev/null)" \
    || { warn "BOLT[$label]: pyperformance not importable in $venv; cannot build a profile — skipping"; return 0; }
  bench_d="$pp/data-files/benchmarks"
  sp="$("$venv" -c 'import site;print(site.getsitepackages()[0])' 2>/dev/null)" \
    || { warn "BOLT[$label]: could not resolve site-packages for $venv; skipping"; return 0; }

  local prof="$WORKDIR/bolt/$label"
  rm -rf "$prof"; mkdir -p "$prof"
  local inst="$bin.inst"
  local TSPRE; TSPRE="$(taskset_prefix)"

  # 1) Instrument. --instrumentation-file-append-pid gives one .fdata per process
  #    so multiple benchmark runs accumulate instead of overwriting.
  log "BOLT[$label]: instrumenting $bin"
  if ! llvm-bolt "$bin" -instrument \
        --instrumentation-file="$prof/prof.fdata" \
        --instrumentation-file-append-pid \
        -o "$inst" >"$LOGDIR/bolt_${label}_instrument.log" 2>&1; then
    warn "BOLT[$label]: instrumentation failed (see $LOGDIR/bolt_${label}_instrument.log); original binary kept"
    rm -f "$inst"
    return 0
  fi

  # 2) Run the representative workload with the instrumented binary. CINDERX_DISABLE=1
  #    keeps the profile to the interpreter core; PYTHONPATH exposes pyperf.
  local ran=0 b
  local OLDIFS="$IFS"; IFS=','
  for b in $BOLT_BENCHES; do
    IFS="$OLDIFS"
    local bm="$bench_d/bm_$b/run_benchmark.py"
    if [ ! -f "$bm" ]; then warn "BOLT[$label]: no pyperformance bm_$b for profiling; skipping it"; IFS=','; continue; fi
    log "BOLT[$label]: profiling with bm_$b"
    if env CINDERX_DISABLE=1 PYTHONPATH="$sp" $TSPRE "$inst" "$bm" --worker -l 1 -w 0 -n 1 \
         >"$LOGDIR/bolt_${label}_run_$b.log" 2>&1; then
      ran=$((ran + 1))
    else
      warn "BOLT[$label]: profiling run bm_$b failed (see $LOGDIR/bolt_${label}_run_$b.log)"
    fi
    IFS=','
  done
  IFS="$OLDIFS"

  if [ "$ran" -eq 0 ]; then
    warn "BOLT[$label]: no profiling workload ran; BOLT skipped, original binary kept"
    rm -f "$inst"
    return 0
  fi

  # 3) Merge the per-process profiles into one fdata (merge-fdata ships with BOLT).
  local fdata=( "$prof"/prof.fdata* )
  if [ ! -e "${fdata[0]}" ]; then
    warn "BOLT[$label]: instrumented run produced no .fdata; BOLT skipped, original kept"
    rm -f "$inst"
    return 0
  fi
  local merged="$prof/merged.fdata"
  if command -v merge-fdata >/dev/null 2>&1; then
    if ! merge-fdata "${fdata[@]}" > "$merged" 2>"$LOGDIR/bolt_${label}_merge.log"; then
      warn "BOLT[$label]: merge-fdata failed (see $LOGDIR/bolt_${label}_merge.log); BOLT skipped, original kept"
      rm -f "$inst"
      return 0
    fi
  elif [ "${#fdata[@]}" -eq 1 ]; then
    merged="${fdata[0]}"
  else
    warn "BOLT[$label]: merge-fdata unavailable and ${#fdata[@]} profiles present; using only the first"
    merged="${fdata[0]}"
  fi

  # 4) Optimize using the profile.
  local opt="$bin.bolt"
  log "BOLT[$label]: optimizing (reorder-blocks=ext-tsp, reorder-functions=cdsort, split-functions, split-all-cold)"
  if ! llvm-bolt "$bin" -o "$opt" -data="$merged" \
        -reorder-blocks=ext-tsp -reorder-functions=cdsort \
        -split-functions -split-all-cold -dyno-stats \
        >"$LOGDIR/bolt_${label}_optimize.log" 2>&1; then
    warn "BOLT[$label]: optimization failed (see $LOGDIR/bolt_${label}_optimize.log); original binary kept"
    rm -f "$inst" "$opt"
    return 0
  fi

  # 5) Swap in the optimized binary (backup the original first) and smoke-test it.
  cp -f "$bin" "$bin.prebolt" || { warn "BOLT[$label]: could not back up original; keeping original, skipping swap"; rm -f "$inst" "$opt"; return 0; }
  if ! cp -f "$opt" "$bin"; then
    warn "BOLT[$label]: could not install optimized binary; restoring original"
    cp -f "$bin.prebolt" "$bin"
    rm -f "$inst" "$opt"
    return 0
  fi
  if ! "$py" -c 'import sys; sys.exit(0)' >/dev/null 2>&1; then
    warn "BOLT[$label]: optimized binary fails to run; restoring original"
    cp -f "$bin.prebolt" "$bin"
    rm -f "$inst" "$opt" "$bin.bolt-done"
    return 0
  fi
  : > "$bin.bolt-done"
  rm -f "$inst"
  ok "BOLT[$label]: $bin optimized (original backed up at $bin.prebolt; dyno-stats in $LOGDIR/bolt_${label}_optimize.log)"
  return 0
}

bolt_optimize() {
  [ "$DO_BOLT" -eq 1 ] || return 0
  log "BOLT: post-link optimization of both interpreter binaries (benches=$BOLT_BENCHES)"
  bolt_one "$PY_PREFIX"        "$VPY"        "plain"
  bolt_one "$PY_PREFIX_STATIC" "$VPY_STATIC" "static"
}

###############################################################################
# Suite A: built-in lightweight benchmarks — 8-way.
###############################################################################
run_builtin() {
  [ "$RUN_BUILTIN" -eq 1 ] || { warn "Skipping suite A (built-in)"; return; }
  local bdir; bdir="$(find_bench_dir)"
  [ -n "$bdir" ] && [ -d "$bdir" ] || { warn "cinderx benchmarks dir not found; skipping suite A"; return; }
  log "Suite A: CinderX built-in lightweight benchmarks (8-way), $TRIALS trials"

  local tsv="$RESULTS/builtin.tsv"
  printf 'name\tconfig\ttrial\tstatus\tseconds\n' > "$tsv"
  local TSPRE; TSPRE="$(taskset_prefix)"

  # benchmark|arg : args chosen so steady-state work dominates startup+warmup.
  local MATRIX=( "binary_trees|1" "fannkuch|2" "nbody|1" "richards|50" "spectral_norm|1" )
  for row in "${MATRIX[@]}"; do
    local name="${row%%|*}" arg="${row##*|}"
    local script="$bdir/$name.py"
    [ -f "$script" ] || { warn "missing $script"; continue; }
    local cfgrow
    for cfgrow in "${ALL_CONFIGS[@]}"; do
      local cname kind cenv vpy
      IFS='|' read -r cname kind cenv <<< "$cfgrow"
      vpy="$(config_vpy "$kind")"
      [ -x "$vpy" ] || { warn "venv for config $cname missing ($vpy); skipping"; continue; }

      # The built-in benchmark scripts are downloaded source we can't modify, and
      # they `import cinderx.jit` (+ call cinderx.jit.auto()) unconditionally at
      # module scope. Decide how to satisfy that import for THIS config's
      # interpreter:
      #   - real cinderx importable (static venv builtin, or plain venv with a
      #     dynamic CinderX wheel) -> run as-is; the script's own auto() applies.
      #   - no real cinderx + "off" config -> put the no-op cinderx shim on
      #     PYTHONPATH so the bench runs as a genuine plain-CPython baseline
      #     (off == JIT disabled, which is exactly what the shim yields).
      #   - no real cinderx + non-off config -> a JIT config with no engine
      #     available; SKIP (don't fabricate plain-CPython numbers under a JIT
      #     label). Requires the dynamic CinderX wheel in the plain venv.
      local pypath_extra="" run_note=""
      if "$vpy" -c 'import cinderx.jit' >/dev/null 2>&1; then
        : # real cinderx available; nothing extra needed
      elif [ "${cname##*-}" = "off" ]; then
        pypath_extra="$HELPERS/cinderx_shim"
        run_note=" [cinderx shim: no real cinderx, JIT disabled]"
      else
        for t in $(seq 1 "$TRIALS"); do
          printf '%s\t%s\t%s\tSKIP\t\n' "$name" "$cname" "$t" >> "$tsv"
        done
        warn "$name/$cname SKIPPED: interpreter cannot import cinderx and config needs the JIT (install the dynamic CinderX wheel into the plain venv)"
        continue
      fi

      for t in $(seq 1 "$TRIALS"); do
        local errf="$LOGDIR/builtin_${name}_${cname}_${t}.log"
        local secs
        secs="$( { /usr/bin/time -f '%e' env $cenv ${pypath_extra:+PYTHONPATH="$pypath_extra${PYTHONPATH:+:$PYTHONPATH}"} $TSPRE "$vpy" "$script" "$arg" \
                    >/dev/null 2>"$errf"; } && tail -1 "$errf" | grep -Eo '^[0-9]+\.[0-9]+$' )"
        if [ -n "$secs" ]; then
          printf '%s\t%s\t%s\tOK\t%s\n' "$name" "$cname" "$t" "$secs" >> "$tsv"
          printf '    %-14s %-14s trial %s: %ss%s\n' "$name" "$cname" "$t" "$secs" "$run_note"
        else
          printf '%s\t%s\t%s\tFAIL\t\n' "$name" "$cname" "$t" >> "$tsv"
          warn "$name/$cname trial $t FAILED (see $errf)"
        fi
      done
    done
  done
  "$VPY" "$HELPERS/report_8way.py" "$tsv" \
    "Suite A — CinderX built-in lightweight benchmarks (8-way)" \
    "$RESULTS/REPORT_builtin.md" "$CONFIG_ORDER" >/dev/null
  ok "Suite A report: $RESULTS/REPORT_builtin.md"
}

###############################################################################
# Suite B: fastmark (pyperformance workloads via cinderx) — 8-way.
#
# Unlike the original ON/OFF suite, the CinderX/JIT policy is now driven purely
# by the env-driven sitecustomize (CINDERX_DISABLE / CINDERX_NO_JIT /
# BENCH_JITLIST_DIR / default-auto) so all eight configurations are uniform —
# fastmark's own --cinderx flag is NOT used. The benchmark scripts are the same
# in every config; the difference is which interpreter (plain/static venv) runs
# them and how sitecustomize enables the JIT.
###############################################################################
run_fastmark() {
  [ "$RUN_FASTMARK" -eq 1 ] || { warn "Skipping suite B (fastmark)"; return; }
  local bdir; bdir="$(find_bench_dir)"
  local fm="$bdir/fastmark.py"
  [ -n "$bdir" ] && [ -f "$fm" ] || { warn "fastmark.py not found; skipping suite B"; return; }
  log "Suite B: fastmark (8-way, scale=$FASTMARK_SCALE)"

  # fastmark.py runs ALL of its benchmarks in a SINGLE process and only writes its
  # --json at the very end, so one uncaught exception (e.g. a JIT/runtime bug that
  # only triggers under the JIT, like docutils raising "list.remove(x): x not in
  # list") aborts the whole run and discards EVERY result. We can't patch
  # fastmark.py or the benchmark deps (downloaded source), so instead we run each
  # benchmark in its OWN fastmark process and merge the per-benchmark JSON. A crash
  # then only loses that one benchmark/config, and per-benchmark isolation also
  # avoids cross-benchmark JIT-state pollution. Enumerate the exact set fastmark
  # would run by default (ALL_BENCHMARKS minus EXCLUDED and pyston-only entries).
  local names
  names="$("$VPY" - "$fm" 2>"$LOGDIR/fastmark_enumerate.log" <<'PYEOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("fastmark", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for name, value in mod.ALL_BENCHMARKS.items():
    if len(value) > 0 and name not in mod.EXCLUDED and value[2] != "pyston":
        print(name)
PYEOF
)"
  [ -n "$names" ] || { warn "could not enumerate fastmark benchmarks (see $LOGDIR/fastmark_enumerate.log); skipping suite B"; return; }

  local nbench; nbench="$(printf '%s\n' "$names" | grep -c .)"
  log "  fastmark: $nbench benchmarks × 8 configs, each run in its own process"
  local parts="$RESULTS/fastmark_parts"
  rm -rf "$parts"

  local tsv="$RESULTS/fastmark.tsv"
  printf 'name\tconfig\ttrial\tstatus\tseconds\n' > "$tsv"
  local TSPRE; TSPRE="$(taskset_prefix)"

  local cfgrow
  for cfgrow in "${ALL_CONFIGS[@]}"; do
    local cname kind cenv vpy
    IFS='|' read -r cname kind cenv <<< "$cfgrow"
    vpy="$(config_vpy "$kind")"
    [ -x "$vpy" ] || { warn "venv for config $cname missing ($vpy); skipping"; continue; }
    mkdir -p "$parts/$cname"
    local b pj plog rc nok=0 nfail=0
    log "  fastmark $cname (per-benchmark isolation)"
    while IFS= read -r b; do
      [ -n "$b" ] || continue
      pj="$parts/$cname/$b.json"
      plog="$LOGDIR/fastmark_${cname}_${b}.log"
      env $cenv $TSPRE "$vpy" "$fm" --scale "$FASTMARK_SCALE" --json "$pj" "$b" \
        >"$plog" 2>&1; rc=$?
      if [ "$rc" -eq 0 ] && [ -s "$pj" ]; then
        nok=$((nok + 1))
      else
        nfail=$((nfail + 1)); rm -f "$pj"
        warn "fastmark $cname: benchmark '$b' failed (see $plog)"
      fi
    done <<< "$names"
    ok "fastmark $cname: $nok ok, $nfail failed"

    # Merge this config's per-benchmark JSON fragments and append to the TSV.
    "$VPY" - "$parts/$cname" "$cname" "$tsv" <<'PYEOF' || warn "fastmark $cname merge failed"
import json, sys, os, glob, csv
d, cfg, tsv = sys.argv[1], sys.argv[2], sys.argv[3]
merged = {}
for p in sorted(glob.glob(os.path.join(d, "*.json"))):
    try:
        with open(p) as f:
            data = json.load(f)
    except Exception:
        continue
    # fastmark --json writes {benchmark_name: seconds, ...}.
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (int, float)):
                merged[k] = float(v)
            elif isinstance(v, dict):
                for kk in ("time", "seconds", "min", "mean"):
                    if kk in v:
                        merged[k] = float(v[kk]); break
with open(tsv, "a", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    for n in sorted(merged):
        w.writerow([n, cfg, 1, "OK", f"{merged[n]:.6f}"])
print(f"fastmark {cfg}: {len(merged)} benchmarks parsed")
PYEOF
  done

  # Number of data rows beyond the header.
  if [ "$(wc -l < "$tsv")" -gt 1 ]; then
    "$VPY" "$HELPERS/report_8way.py" "$tsv" \
      "Suite B — fastmark (pyperformance via cinderx) (8-way)" \
      "$RESULTS/REPORT_fastmark.md" "$CONFIG_ORDER" >/dev/null
    ok "Suite B report: $RESULTS/REPORT_fastmark.md"
  else
    warn "fastmark produced no parseable results; see $LOGDIR/fastmark_*.log"
  fi
}

###############################################################################
# Suite C: Static-Python variants vs non-static — 8-way.
###############################################################################
run_static() {
  [ "$RUN_STATIC" -eq 1 ] || { warn "Skipping suite C (static)"; return; }
  local bdir; bdir="$(find_bench_dir)"
  [ -n "$bdir" ] && [ -d "$bdir" ] || { warn "cinderx benchmarks dir not found; skipping suite C"; return; }
  log "Suite C: Static-Python variants vs non-static (8-way), $TRIALS trials"

  local tsv="$RESULTS/static.tsv"
  # name = script (the kind, base vs static, is evident in the script name).
  printf 'name\tconfig\ttrial\tstatus\tseconds\n' > "$tsv"
  local TSPRE; TSPRE="$(taskset_prefix)"

  # family|arg|script|kind  (kind: base = plain Python, static = strict/static loader)
  # NOTE: fannkuch_static.py is intentionally omitted. Its library
  # (cinderx/benchmarks/fannkuch_static_lib.py) does not compile under Static
  # Python in this CinderX/CPython 3.14 combination — the strict/static loader
  # raises a compile-time TypedSyntaxError at `count[r - 1] = r`
  # ("type mismatch: int64 cannot be assigned to dynamic"), so it FAILs every
  # trial both ON and OFF. That is a bug in the downloaded benchmark source, which
  # this script must not modify, so we skip the variant rather than record a
  # guaranteed failure on every run. The fannkuch_static_basic / _basic2 variants
  # below still exercise the Static-Python path for this family. Re-add the line
  # below if upstream fixes fannkuch_static_lib.py.
  local MATRIX=(
    "fannkuch|2|fannkuch.py|base"
    # "fannkuch|2|fannkuch_static.py|static"   # see NOTE above (uncompilable upstream)
    "fannkuch|2|fannkuch_static_basic.py|static"
    "fannkuch|2|fannkuch_static_basic2.py|static"
    "nbody|1|nbody.py|base"
    "nbody|1|nbody_static.py|static"
    "nbody|1|nbody_static_basic.py|static"
    "richards|50|richards.py|base"
    "richards|50|richards_static.py|static"
    "richards|50|richards_static_basic.py|static"
  )
  for row in "${MATRIX[@]}"; do
    IFS='|' read -r family arg script kind <<< "$row"
    local path="$bdir/$script"
    [ -f "$path" ] || { warn "missing $script; skipping"; continue; }
    local cfgrow
    for cfgrow in "${ALL_CONFIGS[@]}"; do
      local cname ckind cenv vpy
      IFS='|' read -r cname ckind cenv <<< "$cfgrow"
      vpy="$(config_vpy "$ckind")"
      [ -x "$vpy" ] || { warn "venv for config $cname missing ($vpy); skipping"; continue; }
      for t in $(seq 1 "$TRIALS"); do
        local errf="$LOGDIR/static_${script%.py}_${cname}_${t}.log"
        local secs
        secs="$( { /usr/bin/time -f '%e' env $cenv $TSPRE "$vpy" "$path" "$arg" \
                    >/dev/null 2>"$errf"; } && tail -1 "$errf" | grep -Eo '^[0-9]+\.[0-9]+$' )"
        if [ -n "$secs" ]; then
          printf '%s\t%s\t%s\tOK\t%s\n' "$script" "$cname" "$t" "$secs" >> "$tsv"
          printf '    %-28s %-14s trial %s: %ss\n' "$script" "$cname" "$t" "$secs"
        else
          printf '%s\t%s\t%s\tFAIL\t\n' "$script" "$cname" "$t" >> "$tsv"
          warn "$script/$cname trial $t FAILED (see $errf)"
        fi
      done
    done
  done
  "$VPY" "$HELPERS/report_8way.py" "$tsv" \
    "Suite C — Static-Python variants vs non-static (8-way)" \
    "$RESULTS/REPORT_static.md" "$CONFIG_ORDER" >/dev/null
  ok "Suite C report: $RESULTS/REPORT_static.md"
}

###############################################################################
# Suite D: pyperformance — 8-way (the four JIT modes × plain/static CPython).
###############################################################################
# Preflight for Suite D: prove CinderX actually engages under the exact PYTHONPATH
# + CINDERX_SITE_DIR + policy env each pyperformance worker will see, BEFORE the
# long run. We launch the *base* interpreter that pyperformance's compat venv
# wraps (python-install / python-install-static) -- which, like the compat venv,
# has no cinderx on its default path -- with the minimal sitedir on PYTHONPATH.
# A tiny probe reports whether cinderx loaded and whether the JIT is enabled, and
# we assert that against each config's intent. $1 = minimal sitedir.
verify_cinderx_engaged() {
  local sitedir="$1"
  log "Suite D preflight: verifying CinderX engagement per config"
  # jit=1 means the JIT will actually compile code (auto-threshold armed OR a jit
  # list is loaded). is_enabled() is unusable here: importing cinderx flips it on
  # even in nojit mode, so nojit and auto would look identical.
  local probe='import sys
loaded = 1 if "cinderx" in sys.modules else 0
jit = 0
if loaded:
    try:
        import cinderx.jit as j
        armed = j.get_compile_after_n_calls() is not None
        listed = bool(j.get_jit_list() or [])
        jit = 1 if (armed or listed) else 0
    except Exception:
        jit = 0
print("PROBE loaded=%d jit=%d" % (loaded, jit))'
  local cfgrow rc=0
  for cfgrow in "${ALL_CONFIGS[@]}"; do
    local cname kind cenv vpy basepy real_sp exp_loaded exp_jit
    IFS='|' read -r cname kind cenv <<< "$cfgrow"
    vpy="$(config_vpy "$kind")"
    [ -x "$vpy" ] || { warn "  $cname: venv missing ($vpy); skipping preflight"; continue; }
    # jitlist configs are skipped in the real loop when no lists exist; match that.
    case "$cenv" in
      BENCH_JITLIST_DIR=*)
        ls "$JITLIST_DIR"/*.jitlist >/dev/null 2>&1 \
          || { warn "  $cname: no jitlists; skipping preflight"; continue; } ;;
    esac
    case "$kind" in
      plain)  basepy="$PY_PREFIX/bin/python3" ;;
      static) basepy="$PY_PREFIX_STATIC/bin/python3" ;;
      *)      warn "  $cname: unknown kind '$kind'"; rc=1; continue ;;
    esac
    [ -x "$basepy" ] || basepy="$vpy"   # fall back to the venv python if needed
    real_sp="$("$vpy" -c 'import site; print(site.getsitepackages()[0])')" \
      || { warn "  $cname: could not resolve site-packages"; rc=1; continue; }
    # Expected engagement per config ('x' = don't-care). jitlist needs a bm_<name>
    # argv to pick a list, which the preflight lacks, so we only assert it loads.
    case "$cname" in
      *-off)     exp_loaded=0; exp_jit=x ;;
      *-nojit)   exp_loaded=1; exp_jit=0 ;;
      *-auto)    exp_loaded=1; exp_jit=1 ;;
      *-jitlist) exp_loaded=1; exp_jit=x ;;
      *)         exp_loaded=x; exp_jit=x ;;
    esac
    local line got_loaded got_jit
    line="$(env $cenv PYTHONPATH="$sitedir" CINDERX_SITE_DIR="$real_sp" \
            "$basepy" -c "$probe" 2>/dev/null)" || true
    got_loaded="$(printf '%s\n' "$line" | sed -n 's/.*loaded=\([0-9]\).*/\1/p')"
    got_jit="$(printf '%s\n' "$line" | sed -n 's/.*jit=\([0-9]\).*/\1/p')"
    if [ -z "$got_loaded" ]; then
      warn "  $cname: probe produced no output (basepy=$basepy) -> FAIL"; rc=1; continue
    fi
    local okl=1 okj=1
    { [ "$exp_loaded" = x ] || [ "$got_loaded" = "$exp_loaded" ]; } || okl=0
    { [ "$exp_jit" = x ]    || [ "$got_jit" = "$exp_jit" ]; }       || okj=0
    if [ "$okl" = 1 ] && [ "$okj" = 1 ]; then
      ok "  $cname: loaded=$got_loaded jit=$got_jit (want loaded=$exp_loaded jit=$exp_jit)"
    else
      warn "  $cname: loaded=$got_loaded jit=$got_jit BUT want loaded=$exp_loaded jit=$exp_jit -> FAIL"
      rc=1
    fi
  done
  [ "$rc" = 0 ] && ok "Preflight passed: CinderX engages as configured" \
                || warn "Preflight detected configs where CinderX does NOT engage as intended"
  return $rc
}

run_pyperf() {
  [ "$RUN_PYPERF" -eq 1 ] || { warn "Skipping suite D (pyperformance)"; return; }
  log "Suite D: pyperformance (8-way), mode=${PYPERF_MODE:-steady}"

  local tag="$RESULTS/pyperf8way"
  local aff_opt=""
  [ -n "$AFFINITY" ] && aff_opt="--affinity $AFFINITY"
  # pyperformance scrubs the child environment down to HOME/PATH plus whatever we
  # list in --inherit-environ. It creates a per-benchmark venv and pip-installs
  # requirements into it, so the proxy/index variables MUST be inherited or that
  # pip cannot reach PyPI (symptom: "No matching distribution found for
  # setuptools>=18.5"). Forward the CinderX policy vars AND the standard
  # proxy/pip-index variables.
  local inherit="CINDERX_DISABLE,CINDERX_NO_JIT,BENCH_JITLIST_DIR,BENCH_JITLIST_FALLBACK"
  inherit="$inherit,http_proxy,https_proxy,no_proxy,HTTP_PROXY,HTTPS_PROXY,NO_PROXY"
  inherit="$inherit,PIP_INDEX_URL,PIP_EXTRA_INDEX_URL,PIP_TRUSTED_HOST,PIP_CACHE_DIR,PIP_CONFIG_FILE"
  # The CinderX policy only takes effect if sitecustomize.py runs in the actual
  # benchmark worker -- but pyperformance runs benchmarks in an isolated compat
  # venv that has neither our sitecustomize nor cinderx. Fix: put a minimal dir
  # containing only sitecustomize.py on PYTHONPATH (so it is imported by the
  # worker) and point CINDERX_SITE_DIR at the config's real venv site-packages so
  # sitecustomize can site.addsitedir() it (appends -> no shadowing; processes
  # .pth -> static cinderx works). PYTHONPATH is already whitelisted by pyperf's
  # create_environ; CINDERX_SITE_DIR + the banner vars must be inherited too.
  inherit="$inherit,CINDERX_SITE_DIR,BENCH_CINDERX_BANNER,BENCH_CINDERX_BANNER_FILE,PYTHONPATH"
  local common="--benchmarks $PYPERF_BENCHES $aff_opt \
    --inherit-environ $inherit --timeout 600"

  # Minimal PYTHONPATH dir: just a copy of sitecustomize.py (nothing else, to
  # avoid putting a full site-packages ahead of the stdlib on sys.path).
  local sitedir="$WORKDIR/pyperf-sitedir"
  mkdir -p "$sitedir"
  cp "$HELPERS/sitecustomize.py" "$sitedir/sitecustomize.py" \
    || die "could not stage sitecustomize.py into $sitedir"
  rm -rf "$sitedir/__pycache__" 2>/dev/null || true

  # Preflight: prove CinderX actually engages per config BEFORE the long run.
  verify_cinderx_engaged "$sitedir" || die "CinderX engagement preflight failed (see above)"

  local cfgrow pairs=()
  for cfgrow in "${ALL_CONFIGS[@]}"; do
    local cname kind cenv vpy
    IFS='|' read -r cname kind cenv <<< "$cfgrow"
    vpy="$(config_vpy "$kind")"
    [ -x "$vpy" ] || { warn "venv for config $cname missing ($vpy); skipping"; continue; }
    # jitlist configs need the generated lists; skip cleanly if absent.
    case "$cenv" in
      BENCH_JITLIST_DIR=*)
        ls "$JITLIST_DIR"/*.jitlist >/dev/null 2>&1 \
          || { warn "No JIT lists available; skipping config $cname"; continue; } ;;
    esac
    local out="${tag}_${cname}.json"
    # Real venv site-packages for this config -> where cinderx + sitecustomize
    # actually live; sitecustomize will site.addsitedir() it in the worker.
    local real_sp
    real_sp="$("$vpy" -c 'import site; print(site.getsitepackages()[0])')" \
      || { warn "could not resolve site-packages for $cname; skipping"; continue; }
    local wlog="$LOGDIR/cinderx_workers_${cname}.log"
    : > "$wlog"
    log "  $cname ${cenv:+($cenv)}"
    env $cenv \
        PYTHONPATH="$sitedir${PYTHONPATH:+:$PYTHONPATH}" \
        CINDERX_SITE_DIR="$real_sp" \
        BENCH_CINDERX_BANNER=1 \
        BENCH_CINDERX_BANNER_FILE="$wlog" \
        "$vpy" -m pyperformance run $PYPERF_MODE $common \
      -o "$out" >"$LOGDIR/pyperf_${cname}.log" 2>&1 || warn "$cname run had errors (see log)"
    [ -f "$out" ] && pairs+=("$cname=$out")
    # Runtime proof from the actual workers (banner appended by sitecustomize).
    if [ -s "$wlog" ]; then
      local nworkers modes
      nworkers="$(wc -l < "$wlog")"
      modes="$(sort -u "$wlog" | sed 's/ pid=[0-9]*//; s/ bench=[^ ]*//' | sort -u \
               | sed 's/^cinderx-banner: //' | paste -sd'; ' -)"
      ok "    workers=$nworkers cinderx-banner[$cname]: $modes"
    else
      warn "    $cname: NO cinderx banner recorded -> sitecustomize did NOT run in workers!"
    fi
  done

  # Build the 8-way TSV from the per-config pyperf JSONs (mean per benchmark) and
  # render the comparison table. Also keep pairwise pyperformance compare tables
  # against the plain-off baseline for the detailed per-benchmark stats.
  local tsv="$RESULTS/pyperf.tsv"
  if [ "${#pairs[@]}" -gt 0 ]; then
    "$VPY" - "$tsv" "${pairs[@]}" <<'PYEOF' || warn "pyperf TSV build failed"
import sys, csv
import pyperf
tsv = sys.argv[1]
rows = []
for pair in sys.argv[2:]:
    cfg, path = pair.split("=", 1)
    try:
        suite = pyperf.BenchmarkSuite.load(path)
    except Exception as e:
        print("pyperf: could not load %s: %r" % (path, e), file=sys.stderr)
        continue
    for b in suite:
        try:
            rows.append((b.get_name(), cfg, b.mean()))
        except Exception:
            pass
with open(tsv, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["name", "config", "trial", "status", "seconds"])
    for name, cfg, mean in rows:
        w.writerow([name, cfg, 1, "OK", f"{mean:.6f}"])
print("pyperf: %d rows across %d configs" % (len(rows), len(sys.argv) - 2))
PYEOF
  fi

  if [ -s "$tsv" ] && [ "$(wc -l < "$tsv")" -gt 1 ]; then
    "$VPY" "$HELPERS/report_8way.py" "$tsv" \
      "Suite D — pyperformance (8-way), mode=${PYPERF_MODE:-steady-state}, benches=$PYPERF_BENCHES" \
      "$RESULTS/REPORT_pyperf8way.md" "$CONFIG_ORDER" >/dev/null
    # Append per-benchmark pyperformance compare tables vs the plain-off baseline.
    {
      echo
      echo "## Pairwise pyperformance compare vs \`plain-off\` baseline"
      local base="${tag}_plain-off.json" other oc
      if [ -f "$base" ]; then
        for cfgrow in "${ALL_CONFIGS[@]}"; do
          IFS='|' read -r oc _ _ <<< "$cfgrow"
          [ "$oc" = "plain-off" ] && continue
          other="${tag}_${oc}.json"
          [ -f "$other" ] || continue
          echo; echo "### plain-off vs $oc"; echo '```'
          "$VPY" -m pyperformance compare "$base" "$other" -O table 2>&1
          echo '```'
        done
      else
        echo; echo "_(plain-off baseline JSON missing; pairwise tables skipped)_"
      fi
    } >> "$RESULTS/REPORT_pyperf8way.md"
    ok "Suite D report: $RESULTS/REPORT_pyperf8way.md"
  else
    warn "pyperformance produced no parseable results; see $LOGDIR/pyperf_*.log"
  fi
}

###############################################################################
# Final summary.
###############################################################################
write_summary() {
  local py_ver py_ver_static dyn_ver static_ver
  py_ver="$([ -x "$PY_PREFIX/bin/python3" ] && "$PY_PREFIX/bin/python3" -V 2>&1 || echo unknown)"
  py_ver_static="$([ -x "$PY_PREFIX_STATIC/bin/python3" ] && "$PY_PREFIX_STATIC/bin/python3" -V 2>&1 || echo unknown)"
  dyn_ver="$([ -n "$VPY" ] && [ -x "$VPY" ] && "$VPY" -m pip show cinderx 2>/dev/null | awk '/^Version:/{print $2}' || echo unknown)"
  static_ver="$([ -d "$SRC_CINDERX/.git" ] && git -C "$SRC_CINDERX" describe --always 2>/dev/null || echo "$CINDERX_TAG")"
  {
    echo "# CinderX benchmark run — SUMMARY (8-way)"
    echo
    echo "- **Plain CPython:** $py_ver (PGO+LTO build at \`$PY_PREFIX\`)"
    echo "- **Static CPython:** $py_ver_static (builtin _cinderx at \`$PY_PREFIX_STATIC\`)"
    echo "- **Dynamic CinderX:** ${dyn_ver:-unknown} (wheel in \`$VENV\`)"
    echo "- **Static CinderX:** ${static_ver:-unknown} (builtin _cinderx + PythonLib in \`$VENV_STATIC\`)"
    echo "- **Affinity:** ${AFFINITY:-none}   **Trials:** $TRIALS   **pyperf mode:** ${PYPERF_MODE:-steady-state}"
    if [ "$DO_BOLT" -eq 1 ]; then
      local bolt_plain="no" bolt_static="no"
      [ -f "$(readlink -f "$PY_PREFIX/bin/python3" 2>/dev/null).bolt-done" ] 2>/dev/null && bolt_plain="yes"
      [ -f "$(readlink -f "$PY_PREFIX_STATIC/bin/python3" 2>/dev/null).bolt-done" ] 2>/dev/null && bolt_static="yes"
      echo "- **BOLT:** enabled (profile benches=\`$BOLT_BENCHES\`) — plain optimized: $bolt_plain, static optimized: $bolt_static"
    else
      echo "- **BOLT:** disabled (pass \`--bolt\` to enable post-link binary-layout optimization)"
    fi
    echo "- **Generated:** $(date)"
    echo
    echo "## Configurations"
    echo "Every suite runs each of these eight, controlled by env-driven sitecustomize:"
    echo
    echo "| # | config | interpreter | CinderX / JIT mode |"
    echo "|---|---|---|---|"
    echo "| 1 | plain-off       | plain  | not imported (CINDERX_DISABLE=1) |"
    echo "| 2 | dyn-nojit       | plain  | imported, JIT off (CINDERX_NO_JIT=1) |"
    echo "| 3 | dyn-auto        | plain  | cinderx.jit.auto() |"
    echo "| 4 | dyn-jitlist     | plain  | per-benchmark JIT lists |"
    echo "| 5 | static-off      | static | not imported (CINDERX_DISABLE=1) |"
    echo "| 6 | static-nojit    | static | imported, JIT off (CINDERX_NO_JIT=1) |"
    echo "| 7 | static-auto     | static | cinderx.jit.auto() |"
    echo "| 8 | static-jitlist  | static | per-benchmark JIT lists |"
    echo
    echo "## Reports"
    for r in REPORT_builtin REPORT_fastmark REPORT_static REPORT_pyperf8way; do
      [ -f "$RESULTS/$r.md" ] && echo "- [${r#REPORT_}]($r.md)"
    done
    echo
    echo "## Method (per suite, all 8-way; baseline column = plain-off)"
    echo "- **A. Built-in:** lightweight cinderx benchmarks, min wall-clock over $TRIALS trials."
    echo "- **B. fastmark:** pyperformance workloads via cinderx/benchmarks/fastmark.py (env-driven, scale=$FASTMARK_SCALE)."
    echo "- **C. Static variants:** plain vs Static-Python scripts (static/strict compile runs even when JIT off)."
    echo "- **D. pyperformance:** env-driven sitecustomize, benches=\`$PYPERF_BENCHES\`."
  } > "$RESULTS/SUMMARY.md"
  ok "Top-level summary: $RESULTS/SUMMARY.md"
}

###############################################################################
# Main.
###############################################################################
main() {
  log "cinderx-benchmark.sh starting (workdir=$WORKDIR)"
  preflight
  write_helpers
  build_cpython          # plain CPython  -> $PY_PREFIX
  build_static_cpython   # static CPython (builtin _cinderx) -> $PY_PREFIX_STATIC
  make_venvs             # creates/locates both VPY (plain) and VPY_STATIC (static)
  # If we skipped venv creation but need the python paths, locate them.
  [ -n "$VPY" ]        || { [ -x "$VENV/bin/python" ]        && VPY="$VENV/bin/python"; }
  [ -n "$VPY_STATIC" ] || { [ -x "$VENV_STATIC/bin/python" ] && VPY_STATIC="$VENV_STATIC/bin/python"; }
  [ -n "$VPY" ]        || die "No usable plain venv python; run without --skip-venv first"
  [ -n "$VPY_STATIC" ] || die "No usable static venv python; run without --skip-venv first"
  # Ensure sitecustomize is present in both venvs (cheap, idempotent) even on --skip-venv.
  if [ -f "$HELPERS/sitecustomize.py" ]; then
    install_sitecustomize "$VPY"
    install_sitecustomize "$VPY_STATIC"
  fi
  # The static venv needs the CinderX source for its PythonLib .pth; the suites
  # also run benchmark scripts from the source tree (not shipped in the wheel).
  ensure_cinderx_source
  install_cinderx            # dynamic wheel into the plain venv (configs 2-4)
  install_cinderx_pythonlib  # PythonLib .pth into the static venv (configs 6-8)
  gen_jitlists               # generated once with the plain venv; shared by both
  bolt_optimize              # (opt-in --bolt) BOLT-optimize both interpreters in place
  run_builtin
  run_fastmark
  run_static
  run_pyperf
  write_summary
  ok "All done. See $RESULTS/SUMMARY.md"
}

main "$@"
