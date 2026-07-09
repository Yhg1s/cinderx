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
#   3. Create THREE benchmark virtualenvs and install pyperformance + the fastmark
#      benchmark dependencies in each. The dynamic and static linkages build
#      CinderX from source (with LTO); the third is the pre-built PyPI wheel:
#         - plain venv:  CinderX built from source into a dynamic _cinderx.so (LTO)
#         - static venv: pure-Python cinderx package (PythonLib) via a .pth file
#                        backed by the builtin _cinderx compiled in step 2 (LTO)
#         - pip venv:    SAME plain interpreter, but CinderX installed as the
#                        pre-built PyPI wheel via `pip install cinderx` (LTO+PGO).
#                        REQUIRED by default (a failed install aborts the run);
#                        pass --skip-pip to omit it (e.g. offline).
#   4. Generate per-benchmark JIT lists once (shared by all interpreters).
#   5. Run four benchmark suites, each across TWELVE configurations — the cross
#      product of {plain, static, pip} CinderX and four JIT modes (off / no-jit /
#      auto / per-benchmark jit-list):
#         1. plain-off       5. static-off       9.  pip-off
#         2. dyn-nojit       6. static-nojit     10. pip-nojit
#         3. dyn-auto        7. static-auto      11. pip-auto
#         4. dyn-jitlist     8. static-jitlist   12. pip-jitlist
#      (With --skip-pip the pip column is dropped and it is an 8-way run.)
#      The four suites are: A. CinderX built-in lightweight benchmarks,
#      B. fastmark, C. Static-Python variants, D. pyperformance.
#   6. Emit an N-way comparison table / markdown report for every suite. The pip
#      configs (LTO+PGO) vs the source-built ones (LTO only) isolate the PGO effect.
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
CINDERX_TAG="${CINDERX_TAG:-main}"             # CinderX git tag/branch to build (CinderX is always built from source)
CINDERX_CC="${CINDERX_CC:-}"                    # override C compiler for CPython + CinderX (default: auto-detect); CLI: --cc
CINDERX_CXX="${CINDERX_CXX:-}"                  # override C++ compiler for CPython + CinderX (default: auto-detect); CLI: --cxx
# Where each compiler override came from, for validation error messages. Seeded
# from the env vars now; the CLI parser rewrites these to "--cc"/"--cxx" when the
# flags are used (CLI beats env). Empty => not explicitly provided => auto-detect.
CC_ORIGIN=""; [ -n "$CINDERX_CC" ] && CC_ORIGIN="env var CINDERX_CC"
CXX_ORIGIN=""; [ -n "$CINDERX_CXX" ] && CXX_ORIGIN="env var CINDERX_CXX"

WORKDIR="${WORKDIR:-}"                          # REQUIRED: root for everything. No default — pass --workdir DIR (or set $WORKDIR).
JOBS="${JOBS:-$(nproc)}"                       # make -j parallelism
AFFINITY="${AFFINITY:-}"                       # CPU core range for taskset, e.g. "8-11" (empty = no pinning)
TRIALS="${TRIALS:-3}"                          # repeats for built-in / static suites (min reported)
PYPERF_MODE="${PYPERF_MODE:---fast}"           # --fast or "" (empty = full steady-state)
REGEN_JITLISTS="${REGEN_JITLISTS:-0}"          # 1 = regenerate JIT lists even if cached
JIT_THRESHOLD="${JIT_THRESHOLD:-2}"            # gen_jitlist hot-function threshold (2=canonical)
JIT_BUDGET="${JIT_BUDGET:-3.0}"               # gen_jitlist workload budget seconds

# Free-threaded (PEP 703, no-GIL) build. When 1, CPython is configured with
# --disable-gil and the CinderX cmake build gets free_threading=1. A free-threaded
# build is a distinct ABI, so its WORKDIR-derived paths get a "-freethreading"
# suffix (see below) to keep it isolated from a GIL-enabled build in the same root.
FREE_THREADING="${FREE_THREADING:-0}"          # 1 = build free-threaded CPython (--disable-gil); CLI: --free-threading

# Which phases to run (all on by default). Disabled via --skip-* flags.
DO_BUILD_CPYTHON=1
DO_INSTALL_CINDERX=1
DO_VENV=1
DO_JITLISTS=1
RUN_BUILTIN=1
RUN_FASTMARK=1
RUN_STATIC=1
RUN_PYPERF=1

# Third "pip" venv: same plain interpreter, but CinderX installed as the pre-built
# PyPI wheel (LTO+PGO) via `pip install cinderx`, giving a 3-way linkage compare
# (pip-wheel LTO+PGO vs source-built LTO vs static-builtin LTO) in every run.
# REQUIRED by default: if the wheel can't be installed the run FAILS. Pass
# --skip-pip (DO_PIP=0) to omit this venv entirely (e.g. offline environments).
DO_PIP="${DO_PIP:-1}"

# ---------------------------------------------------------------------------
# pyperformance benchmark PRESETS for suite D.
#
# --pyperf-benches accepts either a concrete pyperformance selection (a
# comma-separated list of names/groups, "all", negative excludes, ...  — passed
# straight through to pyperformance) OR one of the named presets below, which are
# expanded to a fixed benchmark list before use (see resolve_pyperf_preset):
#
#   quick     10 high-signal benchmarks (5 prior JIT winners + 5 prior
#             regressions): fast enough to iterate, historically the default.
#   reliable  31-benchmark curated low-noise set from the pyperf noise analysis
#             (median within-config CV < 3% across the 8-way BOLT/LTO run),
#             spanning 11 workload types and mixing strong JIT winners with
#             JIT-neutral controls. ~4x smaller than the full ~121-bench suite
#             for meaningful signal at 3-5 trials. Alias: "curated".
#   all       every pyperformance benchmark (handled by pyperformance itself).
#
# The "reliable" list comes from the noise analysis in
# ~/tmp/cinderx-benchmarking-clang-bolt-lto-full-source-builds-2/results/
# (CURATED_SET.md / NOISE_ANALYSIS.md).
PYPERF_PRESET_QUICK="richards,richards_super,spectral_norm,chaos,deltablue,fannkuch,raytrace,generators,go,nqueens"
PYPERF_PRESET_RELIABLE="ascii85_small,async_tree_eager,base32_small,base85_small,bench_thread_pool,chameleon,chaos,connected_components,coroutines,docutils,dulwich_log,float,generators,genshi_text,mdp,nbody,nqueens,pickle_pure_python,pidigits,regex_effbot,regex_v8,richards,scimark_fft,shortest_path,spectral_norm,sqlglot_v2_normalize,sympy_str,tomli_loads,unpack_sequence,xdsl_constant_fold,xml_etree_generate"

# Default suite-D selection is the "quick" preset. Override with --pyperf-benches
# (a list, "all", or a preset name) or --pyperf-reliable (shorthand for the
# curated reliable preset).
PYPERF_BENCHES="${PYPERF_BENCHES:-quick}"

# Expand a preset name in $PYPERF_BENCHES to its concrete benchmark list. Only the
# script's own preset names (quick / reliable / curated) are intercepted; anything
# else (comma lists, "all", pyperformance group names, negative excludes) is left
# untouched so pyperformance's native selection still works.
resolve_pyperf_preset() {
  case "$PYPERF_BENCHES" in
    quick)            PYPERF_BENCHES="$PYPERF_PRESET_QUICK" ;;
    reliable|curated) PYPERF_BENCHES="$PYPERF_PRESET_RELIABLE" ;;
  esac
}

# fastmark work-scale factor (lower = faster; 100 is fastmark's default).
FASTMARK_SCALE="${FASTMARK_SCALE:-100}"

# BOLT post-link optimization (opt-in via --bolt) using CPython's own built-in
# --enable-bolt support. BOLT rewrites the interpreter binary's code layout from
# profile data (basic-block/function reordering, cold splitting), directly
# targeting the ±10% binary-layout effect the foobar7/8 analysis measured between
# the plain and static builds. Off by default: it adds build time and requires
# llvm-bolt + merge-fdata on PATH. When enabled, build_cpython passes --enable-bolt
# to ./configure, which rewires `make` into CPython's bolt-opt pipeline: build the
# PGO+LTO python first, then instrument it with llvm-bolt, TRAIN with the full
# CPython regression suite (PROFILE_TASK, instrumentation-based — no perf/LBR
# needed), and apply the merged profile. configure also auto-injects the required
# link/compile flags (-Wl,--emit-relocs, -fno-pie/-no-pie, and
# -fno-reorder-blocks-and-partition) and skips the eval loop (-skip-funcs) so the
# computed-goto interpreter core is not miscompiled. See build_cpython() and the
# static-build re-BOLT step for the flow.
DO_BOLT="${DO_BOLT:-0}"

# ---------------------------------------------------------------------------
# Derived paths.
# ---------------------------------------------------------------------------
SRC_CPYTHON="$WORKDIR/cpython"
SRC_CINDERX="$WORKDIR/cinderx"
PY_PREFIX="$WORKDIR/python-install"                # plain CPython (no CinderX linked in)
PY_PREFIX_STATIC="$WORKDIR/python-install-static"  # static CPython (builtin _cinderx)
VENV="$WORKDIR/venv"                               # plain venv (dynamic CinderX built from source)
VENV_STATIC="$WORKDIR/venv-static"                 # static venv (PythonLib via .pth)
VENV_PIP="$WORKDIR/venv-pip"                       # pip venv (pre-built CinderX wheel from PyPI)
RESULTS="$WORKDIR/results"
JITLIST_DIR="$WORKDIR/jitlists/lists"
HELPERS="$WORKDIR/helpers"
LOGDIR="$RESULTS/logs"
STATIC_BUILD_DIR="$WORKDIR/cinderx-static-build"   # cmake build tree for the CinderX .a archives

# Populated after the venvs exist.
VPY=""           # plain venv python
VPY_STATIC=""    # static venv python
VPY_PIP=""       # pip venv python (pre-built PyPI CinderX wheel)

# Populated by detect_toolchain(): a single C++20 toolchain used to build BOTH
# CPython (./configure CC=/CXX=) and the CinderX archives, so their libstdc++/ABI
# match.
TOOLCHAIN_CC=""; TOOLCHAIN_CXX=""

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
across TWELVE configurations (source-dynamic / static-builtin / PyPI-wheel
linkage × 4 JIT modes). Use --skip-pip to drop the wheel linkage (8-way).

USAGE:
  bash cinderx-benchmark.sh --workdir DIR [OPTIONS]

  --workdir is REQUIRED (there is no default work directory). It runs the entire
  pipeline: build the plain CPython (PGO+LTO),
  relink a second "static" CPython with _cinderx built in, create the benchmark
  venvs (dynamic CinderX built from source + static PythonLib + a pip-installed
  PyPI wheel unless --skip-pip), generate JIT lists, then run all four suites
  across all configurations and write reports. The dynamic and static linkages
  are always built from source (with LTO); the third "pip" linkage is the
  pre-built PyPI wheel (LTO+PGO), which isolates the PGO effect.

THE TWELVE CONFIGURATIONS (every suite runs each one; 9-12 dropped by --skip-pip):
  1. plain-off        plain CPython, CinderX not imported  (CINDERX_DISABLE=1)
  2. dyn-nojit        plain CPython, dynamic CinderX imported, JIT not enabled
  3. dyn-auto         plain CPython, dynamic CinderX, cinderx.jit.auto()
  4. dyn-jitlist      plain CPython, dynamic CinderX, per-benchmark JIT lists
  5. static-off       static CPython (builtin _cinderx), not imported
  6. static-nojit     static CPython, CinderX imported, JIT not enabled
  7. static-auto      static CPython, cinderx.jit.auto()
  8. static-jitlist   static CPython, per-benchmark JIT lists
  9. pip-off          plain CPython + PyPI wheel (LTO+PGO), not imported
  10. pip-nojit       plain CPython + PyPI wheel, imported, JIT not enabled
  11. pip-auto        plain CPython + PyPI wheel, cinderx.jit.auto()
  12. pip-jitlist     plain CPython + PyPI wheel, per-benchmark JIT lists

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
  --skip-pip             Omit the third "pip" venv (configs 9-12) entirely, making
                         every suite 8-way again. By DEFAULT the pip venv is
                         REQUIRED: the pre-built CinderX wheel is installed from
                         PyPI (`pip install cinderx`) and a failed install ABORTS
                         the run (there is no graceful offline skip). Pass this
                         flag in offline environments or to skip the PGO wheel.
  --only-bench           Shorthand: skip all build/install/venv phases, just
                         run the benchmark suites against existing installs.
  --regen-jitlists       Force regeneration of JIT lists even if cached.

SOURCES / VERSIONS:
  --cpython-tag TAG      CPython git tag/branch to build      (default: v3.14.5)
  --cpython-repo URL     CPython git remote                   (default: github.com/python/cpython)
  --cinderx-repo URL     CinderX git remote (used for BOTH the static and dynamic
                        source builds).                       (default: github.com/facebookincubator/cinderx)
  --cinderx-tag TAG      CinderX git tag/branch to build.     (default: main)

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
  --pyperf-benches SEL   pyperformance selection for suite D: a comma-separated
                        list of benchmark/group names (optionally with negative
                        "-name" excludes), the literal "all", or one of the named
                        presets below.                        (default: quick)
                          quick     10 high-signal benchmarks (5 prior JIT winners
                                    + 5 prior regressions; the default).
                          reliable  31-benchmark curated low-noise set from the
                                    pyperf noise analysis (median within-config
                                    CV < 3%), 11 workload types, mixing JIT winners
                                    and JIT-neutral controls. Alias: curated.
                          all       every pyperformance benchmark (~121).
  --pyperf-reliable      Shorthand for --pyperf-benches reliable (the curated
                        low-noise 31-benchmark set). The set:
                          ascii85_small, async_tree_eager, base32_small,
                          base85_small, bench_thread_pool, chameleon, chaos,
                          connected_components, coroutines, docutils, dulwich_log,
                          float, generators, genshi_text, mdp, nbody, nqueens,
                          pickle_pure_python, pidigits, regex_effbot, regex_v8,
                          richards, scimark_fft, shortest_path, spectral_norm,
                          sqlglot_v2_normalize, sympy_str, tomli_loads,
                          unpack_sequence, xdsl_constant_fold, xml_etree_generate
  --pyperf-quick         Shorthand for --pyperf-benches quick (the default set).
  --fastmark-scale N     fastmark work scale (lower=faster)   (default: 100)
  --jit-threshold N      gen_jitlist hot threshold            (default: 2)
  --jit-budget SECS      gen_jitlist workload budget          (default: 3.0)

FREE-THREADING (PEP 703 / no-GIL) BUILD (optional):
  --free-threading       Build the free-threaded (no-GIL) CPython: CPython is
                        configured with --disable-gil and the CinderX archives are
                        built with free_threading=1 (matching the interpreter's ABI).
                        OFF by default (a normal GIL-enabled build). A free-threaded
                        build is a distinct interpreter/ABI, so ALL work paths under
                        --workdir get a "-freethreading" suffix (sources, installs,
                        venvs, results) — a free-threaded run is fully isolated from
                        a GIL-enabled run in the same --workdir and never reuses its
                        cached builds. Also settable via FREE_THREADING=1.
  --disable-gil          Alias for --free-threading.

BOLT POST-LINK OPTIMIZATION (optional):
  --bolt                 Build CPython with its own --enable-bolt support so `make`
                        runs the built-in bolt-opt pipeline: after the PGO+LTO build,
                        llvm-bolt instruments the interpreter, TRAINS it with the full
                        CPython regression suite (instrumentation-based — no perf or
                        LBR needed), and applies the merged profile (basic-block +
                        function reordering, cold-code splitting, and -skip-funcs for
                        the eval loop). This directly targets the ±10% binary-layout
                        effect measured between the plain and static builds. OFF by
                        default (adds build time; requires llvm-bolt + merge-fdata on
                        PATH — ./configure --enable-bolt HARD-FAILS if either is
                        missing). configure auto-adds the needed flags
                        (-Wl,--emit-relocs, -fno-pie/-no-pie,
                        -fno-reorder-blocks-and-partition). The plain interpreter is
                        BOLTed by `make`; the static interpreter is re-BOLTed after
                        its `make python` relink via CPython's profile-bolt-stamp
                        target (best-effort — the static build still completes if the
                        re-BOLT fails).
  --no-bolt              Explicitly disable BOLT (the default).

  -h, --help             Show this help and exit.

ENVIRONMENT VARIABLES:
  Every option above has a matching env var (CPYTHON_TAG, CINDERX_TAG,
  WORKDIR, JOBS, AFFINITY, TRIALS, PYPERF_MODE, PYPERF_BENCHES, FASTMARK_SCALE,
  JIT_THRESHOLD, JIT_BUDGET, FREE_THREADING, DO_BOLT, DO_PIP, ...). CLI flags
  take precedence over env vars. DO_PIP=0 is equivalent to --skip-pip (the pip
  venv is created and its wheel installed by default; DO_PIP=1).
  WORKDIR is required: set it via --workdir or the $WORKDIR env var (no default).
  CINDERX_CC / CINDERX_CXX override the auto-detected compiler used for BOTH
  CPython and the CinderX archives (the --cc / --cxx flags take precedence over
  them).

OUTPUT:
  Results, logs and markdown reports are written under  <workdir>/results/ .
  A top-level SUMMARY.md links all per-suite reports (12-way, or 8-way with --skip-pip).

REQUIREMENTS (must be pre-installed on a fresh box):
  git, make, a C++20 compiler (gcc 13+ or clang 18+), cmake + ninja + ar (the
  static interpreter is always built), and the usual CPython build deps (zlib,
  openssl/libssl, libffi, readline, bzip2, lzma, sqlite headers).

EXAMPLES:
  bash cinderx-benchmark.sh --workdir ~/cinderx-bench
  bash cinderx-benchmark.sh --workdir ~/cinderx-bench --affinity 8-11 --trials 5
  bash cinderx-benchmark.sh --workdir ~/cinderx-bench --only-bench --skip-jitlists
  bash cinderx-benchmark.sh --workdir ~/cinderx-bench --pyperf-reliable   # curated low-noise suite D
  bash cinderx-benchmark.sh --workdir ~/cinderx-bench --pyperf-benches reliable --trials 5
  bash cinderx-benchmark.sh --workdir ~/cinderx-bench --bolt   # BOLT both binaries
  CPYTHON_TAG=v3.14.5 bash cinderx-benchmark.sh --workdir ~/cinderx-bench --cinderx-tag main
  bash cinderx-benchmark.sh --workdir ~/cinderx-bench --cinderx-repo /path/to/cinderx --cinderx-tag main
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
    --skip-pip)           DO_PIP=0 ;;
    --only-bench)         DO_BUILD_CPYTHON=0; DO_INSTALL_CINDERX=0; DO_VENV=0 ;;
    --regen-jitlists)     REGEN_JITLISTS=1 ;;
    --free-threading|--disable-gil) FREE_THREADING=1 ;;
    --bolt)               DO_BOLT=1 ;;
    --no-bolt)            DO_BOLT=0 ;;
    --cpython-tag)        CPYTHON_TAG="$2"; shift ;;
    --cpython-repo)       CPYTHON_REPO="$2"; shift ;;
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
    --pyperf-reliable)    PYPERF_BENCHES="reliable" ;;
    --pyperf-quick)       PYPERF_BENCHES="quick" ;;
    --fastmark-scale)     FASTMARK_SCALE="$2"; shift ;;
    --jit-threshold)      JIT_THRESHOLD="$2"; shift ;;
    --jit-budget)         JIT_BUDGET="$2"; shift ;;
    -h|--help)            usage; exit 0 ;;
    *) die "Unknown option: $1  (try --help)" ;;
  esac
  shift
done

# Expand a preset name (quick / reliable / curated) in $PYPERF_BENCHES to its
# concrete benchmark list. No-op for concrete selections and "all".
resolve_pyperf_preset

# --workdir is required (no default): a benchmark run creates large source/build
# trees, so we never pick a directory on the user's behalf.
if [ -z "$WORKDIR" ]; then
  die "--workdir is required (no default). Pass --workdir DIR (or set \$WORKDIR) to choose the
    root for sources/build/venv/results, e.g.  --workdir ~/cinderx-bench . See --help."
fi

# Re-derive paths in case --workdir changed them.
#
# A free-threaded (no-GIL) build is a distinct ABI/interpreter, so it gets its own
# "-freethreading" suffix on every WORKDIR-derived path. This keeps a free-threaded
# run fully isolated from a GIL-enabled run under the same --workdir: separate
# source trees, installs, venvs, and results (no accidental reuse of the other
# variant's cached build). Empty suffix => the default GIL build uses the plain
# path names, so existing workdirs are unaffected.
FT_SUFFIX=""
[ "$FREE_THREADING" -eq 1 ] && FT_SUFFIX="-freethreading"
SRC_CPYTHON="$WORKDIR/cpython$FT_SUFFIX"
SRC_CINDERX="$WORKDIR/cinderx$FT_SUFFIX"
PY_PREFIX="$WORKDIR/python-install$FT_SUFFIX"
PY_PREFIX_STATIC="$WORKDIR/python-install-static$FT_SUFFIX"
VENV="$WORKDIR/venv$FT_SUFFIX"
VENV_STATIC="$WORKDIR/venv-static$FT_SUFFIX"
VENV_PIP="$WORKDIR/venv-pip$FT_SUFFIX"
RESULTS="$WORKDIR/results$FT_SUFFIX"
JITLIST_DIR="$WORKDIR/jitlists$FT_SUFFIX/lists"
HELPERS="$WORKDIR/helpers$FT_SUFFIX"
LOGDIR="$RESULTS/logs"
STATIC_BUILD_DIR="$WORKDIR/cinderx-static-build$FT_SUFFIX"

###############################################################################
# The eight benchmark configurations.
#
# Each entry is  name|venv_kind|env_assignments  where:
#   - venv_kind is "plain" (source-built dynamic _cinderx.so) or "static" (builtin _cinderx).
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

# The third linkage: the pre-built CinderX wheel (LTO+PGO) from PyPI, run on the
# SAME plain interpreter as the dyn configs. Appended only when enabled (default),
# turning the 8-way matrix into a 12-way one. --skip-pip (DO_PIP=0) drops these so
# no suite tries to run them and the reports stay 8-way. The four pip configs
# mirror the dyn configs' JIT-mode env exactly (off / no-jit / auto / jit-list).
NWAY_LABEL="8-way"
if [ "$DO_PIP" -eq 1 ]; then
  CONFIG_ORDER="$CONFIG_ORDER,pip-off,pip-nojit,pip-auto,pip-jitlist"
  ALL_CONFIGS+=(
    "pip-off|pip|CINDERX_DISABLE=1"
    "pip-nojit|pip|CINDERX_NO_JIT=1"
    "pip-auto|pip|"
    "pip-jitlist|pip|BENCH_JITLIST_DIR=$JITLIST_DIR"
  )
  NWAY_LABEL="12-way"
fi

# Resolve a config's venv-kind to its venv python interpreter.
config_vpy() {
  case "$1" in
    plain)  printf '%s\n' "$VENV/bin/python" ;;
    static) printf '%s\n' "$VENV_STATIC/bin/python" ;;
    pip)    printf '%s\n' "$VENV_PIP/bin/python" ;;
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
  else
    warn "No suitable C++20 compiler (gcc 13+ or clang) found; required to build the static CPython"; missing=1
  fi

  # BOLT (opt-in): with --bolt we build CPython using its built-in --enable-bolt
  # support (see build_cpython). ./configure --enable-bolt HARD-FAILS if it cannot
  # find llvm-bolt AND merge-fdata on PATH, so we just verify both exist here and
  # fail early with a clear message rather than surfacing a confusing configure
  # abort later. (The built-in pipeline is instrumentation-based, so perf/perf2bolt
  # are NOT required.)
  if [ "$DO_BOLT" -eq 1 ]; then
    local _bmiss=0
    command -v llvm-bolt   >/dev/null 2>&1 || { warn "--bolt requested but llvm-bolt not found on PATH"; _bmiss=1; }
    command -v merge-fdata >/dev/null 2>&1 || { warn "--bolt requested but merge-fdata not found on PATH"; _bmiss=1; }
    if [ "$_bmiss" -eq 1 ]; then
      die "BOLT requires both llvm-bolt and merge-fdata on PATH (./configure --enable-bolt hard-fails without them). Install a modern LLVM BOLT toolchain (>=16) and re-run, or drop --bolt."
    fi
    ok "BOLT enabled: llvm-bolt ($(llvm-bolt --version 2>&1 | sed -n 's/.*LLVM version/LLVM/p;q')) + merge-fdata found; CPython will build with --enable-bolt"
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
trials (lower = better).

Each non-baseline cell shows the best time in auto-selected units (s / ms / μs)
plus its % difference in wall-clock time vs the baseline column, e.g.
"456 ms (-5.2%)" (negative = faster than baseline, positive = slower). The
baseline column shows just the time. The geomean row shows the geometric mean of
the per-benchmark speedup ratios (baseline / config), computed from the raw
unrounded times so sub-second benchmarks are never flushed to zero first."""
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


def fmt_time(sec):
    """Format seconds with an auto-selected unit and ~3 significant figures, so
    fast sub-second benchmarks never render as '0.00'."""
    if sec >= 1.0:
        val, unit = sec, "s"
    elif sec >= 1e-3:
        val, unit = sec * 1e3, "ms"
    else:
        val, unit = sec * 1e6, "μs"   # microseconds
    if val >= 100:
        num = f"{val:.0f}"
    elif val >= 10:
        num = f"{val:.1f}"
    else:
        num = f"{val:.2f}"
    return f"{num} {unit}"


def fmt_cell(v, b, is_baseline):
    """Cell text: time (+ % time-difference vs baseline for non-baseline cols)."""
    if v is None:
        return "—"
    if is_baseline or not b:
        return fmt_time(v)
    pct = (v - b) / b * 100.0
    return f"{fmt_time(v)} ({pct:+.1f}%)"


header = "| Benchmark | " + " | ".join(configs) + " |"
sep = "|" + "---|" * (len(configs) + 1)
lines = [f"# {title}\n",
         "Cell = best (min) wall-clock time over trials (lower = better), in "
         "auto-selected units (s / ms / μs). "
         f"Baseline = `{baseline}`; non-baseline cells append the % time "
         "difference vs baseline (negative = faster). The geomean row shows "
         "speedup vs the baseline (>1× = faster than baseline).\n",
         header, sep]

# Speedup ratios are accumulated from the raw unrounded times (baseline / config)
# so the geometric mean is correct even for sub-second benchmarks.
ratios = {c: [] for c in configs}
for name in order:
    b = best(name, baseline)
    cells = []
    for c in configs:
        v = best(name, c)
        cells.append(fmt_cell(v, b, c == baseline))
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
  # (the plain venv when the dynamic CinderX has not been built) that import is a
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
  # libstdc++ versions match. configure records CC/CXX in the Makefile, so the
  # subsequent make uses them too.
  detect_toolchain
  local cc_args=()
  [ -n "$TOOLCHAIN_CC" ]  && cc_args+=("CC=$TOOLCHAIN_CC")
  [ -n "$TOOLCHAIN_CXX" ] && cc_args+=("CXX=$TOOLCHAIN_CXX")
  # When BOLT is enabled, use CPython's built-in --enable-bolt. This rewires the
  # subsequent `make` into the bolt-opt pipeline (build the PGO+LTO python, then
  # instrument it, TRAIN with the full regression suite, and apply the merged
  # profile) and auto-injects every flag BOLT needs: -Wl,--emit-relocs (relocation
  # mode / function reordering), -fno-pie -no-pie, and
  # -fno-reorder-blocks-and-partition, plus -skip-funcs for the eval loop. configure
  # records these in the Makefile, so the static `make python` relink
  # (build_static_cpython) inherits the emit-relocs link flag too, letting the
  # static binary be re-BOLTed there. configure HARD-FAILS if llvm-bolt/merge-fdata
  # are missing (preflight already verified both exist).
  local extra_conf=()
  if [ "$DO_BOLT" -eq 1 ]; then
    extra_conf+=("--enable-bolt")
    log "BOLT enabled: configuring CPython with --enable-bolt (make will run the built-in bolt-opt pipeline)"
  fi
  # Free-threaded (PEP 703) build: configure CPython with --disable-gil so the
  # interpreter is built without the GIL. sysconfig then reports Py_GIL_DISABLED=1,
  # which build_static_cpython() reads to build the CinderX archives to match.
  if [ "$FREE_THREADING" -eq 1 ]; then
    extra_conf+=("--disable-gil")
    log "Free-threading enabled: configuring CPython with --disable-gil (no-GIL build)"
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
# dynamically (-lstdc++). Populates TOOLCHAIN_CC / TOOLCHAIN_CXX.
#
# Everything here is generic: no hardcoded toolchain paths, so it works on any
# Linux (Ubuntu/Debian system gcc, Fedora/RHEL gcc-toolset on PATH, clang, etc.).
# Honours CINDERX_CXX / CINDERX_CC overrides in all modes.
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
  # --free-threading forces free_threading=1 for the CinderX archives even if the
  # auto-detect above somehow disagrees (e.g. a reused/plain interpreter). When the
  # plain CPython was built with --disable-gil the two already agree.
  [ "$FREE_THREADING" -eq 1 ] && ft=1

  # Free-threaded CPython installs its headers under include/python<ver>t (the 't'
  # ABI-flag suffix), NOT include/python<ver>, so the wrapper compile's -I must
  # carry the same suffix or it fails with "'Python.h' file not found". Derive the
  # abiflags from $ft so both the GIL and free-threaded builds resolve correctly.
  local pyabi=""
  [ "$ft" -eq 1 ] && pyabi="t"

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
      -DENABLE_LTO=ON \
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
      -I"$PY_PREFIX/include/python$pyver$pyabi" \
      -c "$wrapper_cpp" -o "$wrapper_o" \
      >"$LOGDIR/static_wrapper_compile.log" 2>&1 \
    || die "wrapper compile failed (see $LOGDIR/static_wrapper_compile.log)"

  # Declare _cinderx as a builtin static module in Modules/Setup.local. Because
  # CPython and the CinderX archives are built with the SAME compiler
  # (detect_toolchain), their libstdc++ versions match, so we link the C++ runtime
  # dynamically (-lstdc++). The archives are wrapped in --start-group/--end-group
  # to resolve their circular references; -lz/-lm satisfy CinderX's deps.
  local a
  log "Writing $SRC_CPYTHON/Modules/Setup.local"
  {
    echo "# Auto-generated by cinderx-benchmark.sh — statically link CinderX."
    echo "# Wrapper .o (PyInit__cinderx) compiled out-of-band; archives from CinderX cmake."
    echo "# Archives wrapped in --start-group/--end-group for their circular references."
    printf '_cinderx %s -Wl,--start-group' "$wrapper_o"
    for a in "${archives[@]}"; do printf ' %s' "$a"; done
    printf ' -Wl,--end-group -lstdc++ -lz -lm\n'
  } > "$SRC_CPYTHON/Modules/Setup.local"

  # --- Preserve PGO across the static relink -------------------------------
  # The plain build gets its PGO from CPython's `profile-opt` target, whose final
  # "use" phase passes the profile-use flag on the *make command line*
  #   $(MAKE) build_all CFLAGS_NODIST="$(CFLAGS_NODIST) $(PGO_PROF_USE_FLAG)"
  # The flag is NOT persisted in the generated Makefile. Editing Setup.local forces
  # the relink to recompile the builtin extension modules (config, posixmodule,
  # _io/*, _sre, _datetimemodule, itertoolsmodule, _collectionsmodule,
  # _functoolsmodule, ... ~40 objects); the interpreter core (Python/ceval.o,
  # Objects/*) is NOT recompiled, so it keeps its PGO. But those recompiled modules,
  # built by a bare `make python`, lose -fprofile-*use — the exact regression this
  # fixes (foobar13: 0 occurrences of fprofile in the relink; the plain build had it
  # on every module compile).
  #
  # Re-apply the flag exactly as profile-opt does: append $(PGO_PROF_USE_FLAG) to
  # CFLAGS_NODIST on the `make python` command line. We pass it as a literal make
  # reference (single-quoted so bash does not touch it) so make resolves it to
  # whatever the Makefile defines for THIS toolchain — clang:
  #   -fprofile-instr-use="$(shell pwd)/code.profclangd"   (expands to this tree)
  # gcc:
  #   -fprofile-use -fprofile-correction                   (reads .gcda beside objects)
  # The profile data (code.profclangd / *.gcda) is left in the build tree by the
  # plain build's profile-opt run, so the flag resolves against real data.
  #
  # Guarded on profile-run-stamp (written only after a successful PGO training run)
  # AND a non-empty PGO_PROF_USE_FLAG, so a reused non-optimized tree still relinks
  # cleanly instead of failing on a missing profile.
  local relink_args=() pgo_use_flag="" base_cflags_nodist=""
  pgo_use_flag="$(sed -n 's/^PGO_PROF_USE_FLAG[[:space:]]*=[[:space:]]*//p' "$SRC_CPYTHON/Makefile" | head -1)"
  if [ -f "$SRC_CPYTHON/profile-run-stamp" ] && [ -n "$pgo_use_flag" ]; then
    # $(CFLAGS_NODIST) is empty in a stock CPython Makefile, but preserve any base
    # value defensively, then append the profile-use flag (mirrors profile-opt).
    base_cflags_nodist="$(sed -n 's/^CFLAGS_NODIST[[:space:]]*=[[:space:]]*//p' "$SRC_CPYTHON/Makefile" | head -1)"
    relink_args+=("CFLAGS_NODIST=${base_cflags_nodist:+$base_cflags_nodist }"'$(PGO_PROF_USE_FLAG)')
    log "PGO: static relink will recompile with profile-use flags (PGO_PROF_USE_FLAG=$pgo_use_flag)"
  else
    warn "PGO: no trained profile in $SRC_CPYTHON (profile-run-stamp / PGO_PROF_USE_FLAG missing); static relink will NOT be PGO-optimized"
  fi

  # Relink. Use `make python` (NOT plain `make`, which would redo the full PGO
  # instrument+train pass). Editing Setup.local makes the Makefile regenerate
  # itself on the first invocation; run again so the new config.c/_cinderx links.
  log "Relinking CPython with the builtin _cinderx (make python -j$JOBS)"
  ( cd "$SRC_CPYTHON" && make python -j"$JOBS" "${relink_args[@]}" ) \
      >"$LOGDIR/static_make_python.log" 2>&1 || true
  if ! "$SRC_CPYTHON/python" -c "import sys; sys.exit(0 if '_cinderx' in sys.builtin_module_names else 1)" 2>/dev/null; then
    log "  (re-running make python after Makefile regeneration)"
    ( cd "$SRC_CPYTHON" && make python -j"$JOBS" "${relink_args[@]}" ) \
        >>"$LOGDIR/static_make_python.log" 2>&1 || true
  fi
  "$SRC_CPYTHON/python" -c "import sys; sys.exit(0 if '_cinderx' in sys.builtin_module_names else 1)" 2>/dev/null \
    || die "could not link builtin _cinderx (see $LOGDIR/static_make_python.log)"
  ok "Linked builtin _cinderx"

  # Verify PGO parity: the relink log should now show the profile-use flag on the
  # recompiled objects (the task's success criterion). Soft-checked — a warning,
  # not a failure, so the pipeline still completes if make short-circuited with
  # nothing to recompile.
  if [ "${#relink_args[@]}" -gt 0 ]; then
    if grep -q 'fprofile' "$LOGDIR/static_make_python.log" 2>/dev/null; then
      ok "PGO: static relink recompiled objects with profile-use flags (PGO retained)"
    else
      warn "PGO: expected profile-use flags in the static relink but none found in $LOGDIR/static_make_python.log"
    fi
  fi

  # --- Re-apply BOLT to the static interpreter (opt-in --bolt) --------------
  # --enable-bolt only BOLTs the build-tree python during the plain `make`
  # (bolt-opt). The static `make python` relink above produced a FRESH binary
  # (new config.c + _cinderx.o); that discards the BOLT layout, because BOLT is a
  # post-link rewrite and cannot survive a relink. Re-apply it here using CPython's
  # OWN built-in profile-bolt-stamp target — the exact instrument -> train (full
  # test suite) -> apply pipeline configure wired up — so the static binary reaches
  # BOLT parity with the plain one, still without any manual llvm-bolt code.
  #
  # profile-bolt-stamp restores $(BUILDPYTHON).prebolt (if present) before
  # instrumenting. That stale .prebolt is the PLAIN pre-BOLT binary from the initial
  # build and would clobber our static relink, so we delete it (plus the stamp and
  # any leftover .fdata) first — the target then treats the freshly relinked STATIC
  # binary as pristine. Best-effort: on any failure we restore the pristine static
  # relink (profile-bolt-stamp's own .prebolt backup) so the build never breaks.
  if [ "$DO_BOLT" -eq 1 ]; then
    log "BOLT: re-applying to the static interpreter via CPython's profile-bolt-stamp (train = full test suite; slow)"
    rm -f "$SRC_CPYTHON/python.prebolt" "$SRC_CPYTHON/profile-bolt-stamp" \
          "$SRC_CPYTHON"/python.*.fdata "$SRC_CPYTHON/python.fdata" 2>/dev/null
    if ( cd "$SRC_CPYTHON" && make profile-bolt-stamp -j"$JOBS" ) >"$LOGDIR/static_bolt.log" 2>&1 \
       && "$SRC_CPYTHON/python" -c "import sys; sys.exit(0 if '_cinderx' in sys.builtin_module_names else 1)" 2>/dev/null; then
      ok "BOLT[static]: static interpreter re-BOLTed (see $LOGDIR/static_bolt.log)"
    else
      warn "BOLT[static]: profile-bolt-stamp failed or produced an unusable binary (see $LOGDIR/static_bolt.log)"
      if [ -f "$SRC_CPYTHON/python.prebolt" ]; then
        warn "BOLT[static]: restoring the pristine static relink from python.prebolt"
        cp -f "$SRC_CPYTHON/python.prebolt" "$SRC_CPYTHON/python" \
          || die "BOLT[static]: failed to restore static python from backup"
      fi
      # Whatever remains must still be a working static interpreter.
      "$SRC_CPYTHON/python" -c "import sys; sys.exit(0 if '_cinderx' in sys.builtin_module_names else 1)" 2>/dev/null \
        || die "BOLT[static]: static python is broken and could not be restored (see $LOGDIR/static_bolt.log)"
      warn "BOLT[static]: continuing with the unbolted static interpreter"
    fi
  fi

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

# Pip venv: install the PRE-BUILT CinderX wheel from PyPI (`pip install cinderx`).
# This is the whole point of the third config: the PyPI wheel is built with
# LTO+PGO, whereas our source builds (plain dynamic + static builtin) are LTO-only,
# so the pip configs isolate the PGO effect against the source-built ones.
#
# REQUIRED by default (no graceful offline skip): if PyPI is unreachable or no
# matching pre-built wheel exists for this interpreter, the whole run FAILS. Pass
# --skip-pip (DO_PIP=0) to intentionally omit the pip venv (e.g. offline), in
# which case this function is never reached (main gates it on DO_PIP).
install_cinderx_pip() {
  if [ "$DO_PIP" -eq 0 ]; then
    warn "Skipping pip CinderX wheel install (--skip-pip)"; return
  fi
  [ -n "$VPY_PIP" ] || die "pip venv not ready (make_venvs must run first)"

  # Idempotent: if the wheel is already importable in the pip venv, do nothing.
  # Probe 'import cinderx.jit' (needs the native extension) — the bare package can
  # import without the runtime, so the weaker probe would give a false positive.
  if "$VPY_PIP" -c 'import cinderx.jit' >/dev/null 2>&1; then
    local ever
    ever="$("$VPY_PIP" -m pip show cinderx 2>/dev/null | awk '/^Version:/{print $2}')"
    ok "pip venv already has CinderX ${ever:-unknown} (PyPI wheel); skipping install"
    return
  fi

  log "Installing CinderX from PyPI (pre-built LTO+PGO wheel) into the pip venv"
  # --only-binary :all: forces a real pre-built WHEEL — never an sdist that would
  # compile from source (which would defeat the purpose of testing the PyPI wheel's
  # LTO+PGO). REQUIRED: any failure here (offline, no compatible wheel, ...) aborts
  # the run. The user must pass --skip-pip to opt out of the pip venv.
  "$VPY_PIP" -m pip install --only-binary :all: cinderx \
      >"$LOGDIR/cinderx_pip_install.log" 2>&1 \
    || die "pip install cinderx FAILED (see $LOGDIR/cinderx_pip_install.log). The pip venv is REQUIRED by default — re-run with --skip-pip to omit it (e.g. offline), or restore PyPI/network access. No pre-built cinderx wheel for this interpreter also triggers this error."

  # Verify the wheel actually provides a usable JIT (native _cinderx + package).
  "$VPY_PIP" -c 'import cinderx.jit as j; print("pip cinderx OK")' \
      >>"$LOGDIR/cinderx_verify.log" 2>&1 \
    || die "CinderX PyPI wheel installed but 'import cinderx.jit' failed (see $LOGDIR/cinderx_verify.log); the wheel may be incompatible with this interpreter (ABI/version). Pass --skip-pip to omit the pip venv."

  local ver
  ver="$("$VPY_PIP" -m pip show cinderx 2>/dev/null | awk '/^Version:/{print $2}')"
  ok "CinderX PyPI wheel (LTO+PGO) installed into the pip venv (version ${ver:-unknown})"
}

###############################################################################
# Phase 2: create the benchmark venvs (plain + static, and the pip venv unless
# --skip-pip) and install deps.
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
  # Plain venv (source-built dynamic CinderX goes in later via install_cinderx).
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

  # Pip venv: uses the SAME plain interpreter ($PY_PREFIX) as the plain venv, but
  # the pre-built PyPI CinderX wheel goes in later via install_cinderx_pip. Only
  # created when the pip config is enabled (default); --skip-pip omits it.
  if [ "$DO_PIP" -eq 1 ]; then
    if [ "$DO_VENV" -eq 0 ] && [ -x "$VENV_PIP/bin/python" ]; then
      VPY_PIP="$VENV_PIP/bin/python"; ok "Reusing existing pip venv $VENV_PIP"
    else
      create_one_venv "$PY_PREFIX" "$VENV_PIP" "pip"
      VPY_PIP="$VENV_PIP/bin/python"
    fi
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
# Phase 3: build the dynamic CinderX from source and install it into the PLAIN
# venv (always a source build of _cinderx.so from $SRC_CINDERX with LTO — never a
# pre-built PyPI wheel). The static venv instead gets the pure-Python PythonLib
# via install_cinderx_pythonlib (its native side is the builtin _cinderx in the
# static interpreter, also compiled from source). Both linkages therefore build
# CinderX from the same source tree with LTO enabled.
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

  # Always build the dynamic CinderX from SOURCE — never a pre-built PyPI wheel.
  # The wheel lacks our LTO + -fno-semantic-interposition changes, so it can't be
  # used to test those optimizations on the dynamic build. Building from source
  # here (into _cinderx.so) mirrors the static build, which also compiles CinderX
  # from $SRC_CINDERX, giving a controlled, reproducible A/B between the two
  # linkages. ensure_cinderx_source() (called before us in main) guarantees the
  # checkout exists; --cinderx-repo/--cinderx-tag choose what it points at.
  [ -f "$SRC_CINDERX/setup.py" ] \
    || die "CinderX source not found at $SRC_CINDERX (expected setup.py). ensure_cinderx_source must run first (check the CinderX clone / --cinderx-repo / --cinderx-tag)."

  # Match the toolchain used for CPython + the static CinderX archives so the
  # dynamic _cinderx.so links against a compatible libstdc++/ABI. detect_toolchain
  # is idempotent and populates TOOLCHAIN_CC/TOOLCHAIN_CXX.
  detect_toolchain

  # CINDERX_ENABLE_LTO=1 -> setup.py adds -DENABLE_LTO=ON to its cmake args,
  # matching the static build's -DENABLE_LTO=ON. CC/CXX pin the same compiler
  # detect_toolchain() selected (setup.py honours both when set).
  local build_env=(CINDERX_ENABLE_LTO=1)
  [ -n "$TOOLCHAIN_CC" ]  && build_env+=("CC=$TOOLCHAIN_CC")
  [ -n "$TOOLCHAIN_CXX" ] && build_env+=("CXX=$TOOLCHAIN_CXX")

  log "Building dynamic CinderX from source ($SRC_CINDERX) with LTO into the plain venv"
  "$VPY" -m pip install setuptools \
    >"$LOGDIR/cinderx_setuptools.log" 2>&1 || die "setuptools install failed"
  ( cd "$SRC_CINDERX" \
    && env "${build_env[@]}" "$VPY" -m pip install -e . --no-build-isolation --force-reinstall \
         >"$LOGDIR/cinderx_build.log" 2>&1 ) \
    || die "CinderX source build failed (see $LOGDIR/cinderx_build.log)"

  # Confirm LTO actually reached the dynamic build. setup.py prints
  # "Building with LTO enabled (full LTO)" whenever CINDERX_ENABLE_LTO is set, and
  # the CinderX CMakeLists emits "LTO: Enabled" once -DENABLE_LTO=ON reaches cmake.
  # foobar17 showed the dynamic _cinderx.so can be built WITHOUT LTO if the editable
  # install doesn't inherit the env (the cibuildwheel LTO setting applies only to
  # wheel builds), so surface that here instead of letting it pass silently.
  if grep -Eq 'Building with LTO enabled|LTO: Enabled' "$LOGDIR/cinderx_build.log"; then
    ok "Dynamic CinderX build configured with LTO (confirmed in cinderx_build.log)"
  else
    warn "Could not confirm LTO in the dynamic CinderX build log ($LOGDIR/cinderx_build.log); the _cinderx.so may have been built WITHOUT LTO"
  fi

  # Verify both ON and OFF states work.
  "$VPY" -c 'import cinderx.jit as j; print("cinderx import OK")' \
    >>"$LOGDIR/cinderx_verify.log" 2>&1 || die "cinderx import failed after install"
  local ver
  ver="$("$VPY" -m pip show cinderx 2>/dev/null | awk '/^Version:/{print $2}')"
  ok "CinderX built from source and installed into the plain venv (version ${ver:-unknown})"
}

# Ensure a CinderX *source* checkout exists at $SRC_CINDERX. The static build and
# the static venv's PythonLib need it, and benchmark suites A/B/C run scripts from
# the source tree's cinderx/benchmarks/ directory, which is NOT included in the
# PyPI wheel. Idempotent (build_static_cpython / install_cinderx may already have
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
  # lacks the source-built dynamic CinderX (e.g. a run with --skip-cinderx)
  # makes gen_jitlist.py raise ModuleNotFoundError for EVERY benchmark and
  # silently emit empty lists (the "funcs=0" symptom).
  local genpy="" genlabel=""
  if [ -n "$VPY" ] && "$VPY" -c 'import cinderx.jit' >/dev/null 2>&1; then
    genpy="$VPY"; genlabel="plain venv"
  elif [ -n "$VPY_STATIC" ] && "$VPY_STATIC" -c 'import cinderx.jit' >/dev/null 2>&1; then
    genpy="$VPY_STATIC"; genlabel="static venv"
    warn "Plain venv cannot 'import cinderx'; generating JIT lists with the static venv instead."
    warn "The dynamic configs (dyn-nojit/dyn-auto/dyn-jitlist) will FAIL for the same reason —"
    warn "build the dynamic CinderX into $VENV (re-run without --skip-cinderx to compile it from source)."
  else
    die "Neither venv can 'import cinderx.jit'; cannot generate JIT lists. Build CinderX first (source build into the plain venv and/or the builtin in the static interpreter)."
  fi
  ok "Generating JIT lists with the $genlabel"

  "$genpy" -c 'import pyperformance' 2>/dev/null \
    || die "pyperformance not importable in the $genlabel; cannot generate JIT lists"
  log "Generating JIT lists (threshold=$JIT_THRESHOLD, budget=${JIT_BUDGET}s) -> $JITLIST_DIR"

  # Resolve the benchmark selection ($PYPERF_BENCHES, which may be "all", a group
  # name, or a list with negative excludes) to concrete benchmarks using
  # pyperformance's OWN manifest/selection API, so we generate exactly the set
  # pyperformance would run and — crucially — get each benchmark's real runscript
  # and extra_opts.
  #
  # This is what fixes the METADATA mismatch: a single bm_<group>/run_benchmark.py
  # can define MANY benchmark names, each selected by different extra_opts (e.g.
  # bm_pickle -> pickle / pickle_dict / pickle_list / unpickle / ...; bm_async_tree
  # -> async_tree_io / async_tree_memoization / ...; bm_argparse ->
  # argparse_subparsers). The old code built the path bm_<name>/run_benchmark.py,
  # which does NOT exist for those variant names, so it silently produced no list
  # for any of them; it also never passed extra_opts, so even the base variant's
  # run_benchmark.py aborted on its required positional argument.
  #
  # Emits one TAB-separated "name<TAB>group<TAB>runscript<TAB>opt opt ..." row per
  # benchmark, where group is the bm_<group> directory basename (minus the bm_
  # prefix) — the SAME key the consumer (sitecustomize._detect_benchmark) derives
  # from the worker's run_benchmark.py path, so the list files we write are found.
  local resolved
  resolved="$("$genpy" - "$PYPERF_BENCHES" 2>"$LOGDIR/gjl_resolve.err" <<'PYEOF'
import os, sys
from pyperformance import _manifest
from pyperformance.cli import _select_benchmarks

sel = sys.argv[1]
manifest = _manifest.load_manifest(None)
for bench in _select_benchmarks(sel, manifest):
    runscript = bench.runscript
    if not runscript or not os.path.isfile(runscript):
        continue
    group = os.path.basename(os.path.dirname(runscript))
    if group.startswith("bm_"):
        group = group[3:]
    print("\t".join([bench.name, group, runscript, " ".join(bench.extra_opts or ())]))
PYEOF
)"
  [ -n "$resolved" ] || die "could not resolve benchmark selection '$PYPERF_BENCHES' via pyperformance (see $LOGDIR/gjl_resolve.err)"

  if printf '%s' ",$PYPERF_BENCHES," | grep -qiE ',[[:space:]]*all[[:space:]]*,'; then
    ok "Expanded '$PYPERF_BENCHES' to $(printf '%s\n' "$resolved" | grep -c .) benchmark(s)"
  fi

  # Group the resolved benchmarks by their bm_<group> directory. Benchmarks that
  # share one run_benchmark.py (the METADATA variants) collapse to a single group,
  # and we UNION the hot functions captured across all of that group's variants
  # into one <group>.jitlist. The consumer keys the list off the directory name
  # only (it cannot tell which variant is running), so a per-group union is both
  # the correct file name AND a superset that serves every variant of the group.
  local groups
  groups="$(printf '%s\n' "$resolved" | cut -f2 | awk 'NF && !seen[$0]++')"

  # Track outcomes so a systemic failure (e.g. cinderx not importable) is loud
  # instead of silently leaving a directory full of empty lists.
  local made=0 failed=0 nvariants=0
  local g
  while IFS= read -r g; do
    [ -n "$g" ] || continue
    local jl="$JITLIST_DIR/$g.jitlist"
    local raw="$LOGDIR/gjl_${g}.raw"; : > "$raw"
    local group_rc=0 variant_names=""
    # Run every benchmark NAME that shares this bm_<group>/run_benchmark.py, each
    # with its own extra_opts, and accumulate the hot functions it compiled.
    local name group2 runscript opts
    while IFS=$'\t' read -r name group2 runscript opts; do
      [ "$group2" = "$g" ] || continue
      variant_names="$variant_names $name"
      nvariants=$((nvariants + 1))
      local rc=0
      # $opts is an intentionally-unquoted token list (pyperformance extra_opts,
      # e.g. "pickle_dict", "io", or "--pure-python pickle"). gen_jitlist.py
      # forwards them to run_benchmark.py's argument parser.
      # shellcheck disable=SC2086
      JIT_THRESHOLD="$JIT_THRESHOLD" "$genpy" "$HELPERS/gen_jitlist.py" "$runscript" \
          --budget "$JIT_BUDGET" $opts \
          >"$LOGDIR/gjl_$name.out" 2>"$LOGDIR/gjl_$name.err" || rc=$?
      if [ "$rc" -ne 0 ]; then
        warn "$(printf '%-24s FAILED (rc=%s; see %s)' "$name" "$rc" "$LOGDIR/gjl_$name.err")"
        group_rc=1
      fi
      # Collect real entries (non-comment, non-blank) for the union.
      grep -vE '^[[:space:]]*(#|$)' "$LOGDIR/gjl_$name.out" 2>/dev/null >> "$raw" || true
    done <<< "$resolved"

    local n
    {
      echo "# CinderX JIT list for pyperformance bm_$g"
      echo "# variants (union of hot functions):${variant_names}"
      echo "# gen_jitlist.py threshold=$JIT_THRESHOLD budget=${JIT_BUDGET}s (via $genlabel)"
      echo "# Format: module:qualname (one hot function per line)"
      sort -u "$raw"
    } > "$jl"
    n="$(grep -vcE '^[[:space:]]*(#|$)' "$jl" 2>/dev/null)"; n="${n:-0}"
    if [ "$group_rc" -ne 0 ]; then
      failed=$((failed + 1))
    else
      printf '    %-24s funcs=%-5s variants:%s\n' "$g" "$n" "$variant_names"
      made=$((made + 1))
    fi
  done <<< "$groups"

  if [ "$made" -eq 0 ]; then
    die "JIT-list generation produced no usable lists ($failed failed); see $LOGDIR/gjl_*.err"
  fi
  [ "$failed" -eq 0 ] || warn "$failed JIT list group(s) failed to generate; see $LOGDIR/gjl_*.err"
  ok "JIT lists generated in $JITLIST_DIR ($made group(s), $nvariants benchmark variant(s), $failed failed)"
}

###############################################################################
# Suite A: built-in lightweight benchmarks — 8-way.
###############################################################################
run_builtin() {
  [ "$RUN_BUILTIN" -eq 1 ] || { warn "Skipping suite A (built-in)"; return; }
  local bdir; bdir="$(find_bench_dir)"
  [ -n "$bdir" ] && [ -d "$bdir" ] || { warn "cinderx benchmarks dir not found; skipping suite A"; return; }
  log "Suite A: CinderX built-in lightweight benchmarks ($NWAY_LABEL), $TRIALS trials"

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
      #   - real cinderx importable (static venv builtin, or plain venv with the
      #     source-built dynamic CinderX) -> run as-is; the script's own auto() applies.
      #   - no real cinderx + "off" config -> put the no-op cinderx shim on
      #     PYTHONPATH so the bench runs as a genuine plain-CPython baseline
      #     (off == JIT disabled, which is exactly what the shim yields).
      #   - no real cinderx + non-off config -> a JIT config with no engine
      #     available; SKIP (don't fabricate plain-CPython numbers under a JIT
      #     label). Requires the source-built dynamic CinderX in the plain venv.
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
        warn "$name/$cname SKIPPED: interpreter cannot import cinderx and config needs the JIT (build the dynamic CinderX from source into the plain venv)"
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
    "Suite A — CinderX built-in lightweight benchmarks ($NWAY_LABEL)" \
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
  log "Suite B: fastmark ($NWAY_LABEL, scale=$FASTMARK_SCALE)"

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
      "Suite B — fastmark (pyperformance via cinderx) ($NWAY_LABEL)" \
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
  log "Suite C: Static-Python variants vs non-static ($NWAY_LABEL), $TRIALS trials"

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
    "Suite C — Static-Python variants vs non-static ($NWAY_LABEL)" \
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
      # pip configs run on the SAME plain interpreter as the dyn configs (CinderX
      # comes from the pip-installed PyPI wheel, not a different build), so their
      # base python is the plain install — identical to "plain".
      pip)    basepy="$PY_PREFIX/bin/python3" ;;
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
  log "Suite D: pyperformance ($NWAY_LABEL), mode=${PYPERF_MODE:-steady}"

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
      "Suite D — pyperformance ($NWAY_LABEL), mode=${PYPERF_MODE:-steady-state}, benches=$PYPERF_BENCHES" \
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
  local py_ver py_ver_static dyn_ver static_ver pip_ver
  py_ver="$([ -x "$PY_PREFIX/bin/python3" ] && "$PY_PREFIX/bin/python3" -V 2>&1 || echo unknown)"
  py_ver_static="$([ -x "$PY_PREFIX_STATIC/bin/python3" ] && "$PY_PREFIX_STATIC/bin/python3" -V 2>&1 || echo unknown)"
  dyn_ver="$([ -n "$VPY" ] && [ -x "$VPY" ] && "$VPY" -m pip show cinderx 2>/dev/null | awk '/^Version:/{print $2}' || echo unknown)"
  static_ver="$([ -d "$SRC_CINDERX/.git" ] && git -C "$SRC_CINDERX" describe --always 2>/dev/null || echo "$CINDERX_TAG")"
  pip_ver="$([ "$DO_PIP" -eq 1 ] && [ -n "$VPY_PIP" ] && [ -x "$VPY_PIP" ] && "$VPY_PIP" -m pip show cinderx 2>/dev/null | awk '/^Version:/{print $2}' || echo unknown)"
  {
    echo "# CinderX benchmark run — SUMMARY ($NWAY_LABEL)"
    echo
    echo "- **Plain CPython:** $py_ver (PGO+LTO build at \`$PY_PREFIX\`)"
    echo "- **Static CPython:** $py_ver_static (builtin _cinderx at \`$PY_PREFIX_STATIC\`)"
    echo "- **Dynamic CinderX:** ${dyn_ver:-unknown} (built from source, LTO-only, into \`$VENV\`)"
    echo "- **Static CinderX:** ${static_ver:-unknown} (builtin _cinderx + PythonLib, LTO-only, in \`$VENV_STATIC\`)"
    if [ "$DO_PIP" -eq 1 ]; then
      echo "- **Pip CinderX:** ${pip_ver:-unknown} (pre-built PyPI wheel, LTO+PGO, on the plain interpreter in \`$VENV_PIP\`)"
    else
      echo "- **Pip CinderX:** skipped (\`--skip-pip\`)"
    fi
    echo "- **Affinity:** ${AFFINITY:-none}   **Trials:** $TRIALS   **pyperf mode:** ${PYPERF_MODE:-steady-state}"
    if [ "$DO_BOLT" -eq 1 ]; then
      # Report whether each interpreter actually carries a BOLTed layout. Built-in
      # BOLT (make bolt-opt for plain; profile-bolt-stamp re-BOLT for static) adds
      # .text.bolt / .bolt.org.text / .text.hot / .text.cold sections; detect them
      # with readelf/llvm-readelf when available, else just report "enabled".
      # Capture-then-match (here-string) avoids the pipefail+SIGPIPE false negative.
      local bolt_plain="enabled" bolt_static="enabled" _re="" _pb _sb _secs
      if command -v readelf >/dev/null 2>&1; then _re="readelf"
      elif command -v llvm-readelf >/dev/null 2>&1; then _re="llvm-readelf"; fi
      if [ -n "$_re" ]; then
        bolt_plain="no"; bolt_static="no"
        _pb="$(readlink -f "$PY_PREFIX/bin/python3" 2>/dev/null)"
        _sb="$(readlink -f "$PY_PREFIX_STATIC/bin/python3" 2>/dev/null)"
        if [ -n "$_pb" ]; then _secs="$("$_re" -S "$_pb" 2>/dev/null)"; grep -qE '\.(text\.bolt|bolt\.org\.text|text\.hot|text\.cold)' <<<"$_secs" && bolt_plain="yes"; fi
        if [ -n "$_sb" ]; then _secs="$("$_re" -S "$_sb" 2>/dev/null)"; grep -qE '\.(text\.bolt|bolt\.org\.text|text\.hot|text\.cold)' <<<"$_secs" && bolt_static="yes"; fi
      fi
      echo "- **BOLT:** enabled via CPython \`--enable-bolt\` (built-in bolt-opt; train = full test suite) — plain BOLTed: $bolt_plain, static re-BOLTed: $bolt_static"
    else
      echo "- **BOLT:** disabled (pass \`--bolt\` to build CPython with \`--enable-bolt\`)"
    fi
    echo "- **Generated:** $(date)"
    echo
    echo "## Configurations"
    if [ "$DO_PIP" -eq 1 ]; then
      echo "Every suite runs each of these twelve, controlled by env-driven sitecustomize."
      echo "The three linkages isolate the optimization effects: source-built dynamic"
      echo "(LTO only) and static-builtin (LTO only) vs the PyPI wheel (LTO **+ PGO**)."
    else
      echo "Every suite runs each of these eight, controlled by env-driven sitecustomize:"
    fi
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
    if [ "$DO_PIP" -eq 1 ]; then
      echo "| 9  | pip-off         | pip (plain interp) | not imported (CINDERX_DISABLE=1) |"
      echo "| 10 | pip-nojit       | pip (plain interp) | imported, JIT off (CINDERX_NO_JIT=1) |"
      echo "| 11 | pip-auto        | pip (plain interp) | cinderx.jit.auto() |"
      echo "| 12 | pip-jitlist     | pip (plain interp) | per-benchmark JIT lists |"
    fi
    echo
    echo "## Reports"
    for r in REPORT_builtin REPORT_fastmark REPORT_static REPORT_pyperf8way; do
      [ -f "$RESULTS/$r.md" ] && echo "- [${r#REPORT_}]($r.md)"
    done
    echo
    echo "## Method (per suite, all $NWAY_LABEL; baseline column = plain-off)"
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
  make_venvs             # creates/locates VPY (plain), VPY_STATIC (static), VPY_PIP (pip unless --skip-pip)
  # If we skipped venv creation but need the python paths, locate them.
  [ -n "$VPY" ]        || { [ -x "$VENV/bin/python" ]        && VPY="$VENV/bin/python"; }
  [ -n "$VPY_STATIC" ] || { [ -x "$VENV_STATIC/bin/python" ] && VPY_STATIC="$VENV_STATIC/bin/python"; }
  [ -n "$VPY" ]        || die "No usable plain venv python; run without --skip-venv first"
  [ -n "$VPY_STATIC" ] || die "No usable static venv python; run without --skip-venv first"
  if [ "$DO_PIP" -eq 1 ]; then
    [ -n "$VPY_PIP" ] || { [ -x "$VENV_PIP/bin/python" ] && VPY_PIP="$VENV_PIP/bin/python"; }
    [ -n "$VPY_PIP" ] || die "No usable pip venv python; run without --skip-venv first (or pass --skip-pip to omit the pip config)"
  fi
  # Ensure sitecustomize is present in every venv (cheap, idempotent) even on --skip-venv.
  if [ -f "$HELPERS/sitecustomize.py" ]; then
    install_sitecustomize "$VPY"
    install_sitecustomize "$VPY_STATIC"
    [ "$DO_PIP" -eq 1 ] && [ -n "$VPY_PIP" ] && install_sitecustomize "$VPY_PIP"
  fi
  # The static venv needs the CinderX source for its PythonLib .pth; the suites
  # also run benchmark scripts from the source tree (not shipped in the wheel).
  ensure_cinderx_source
  install_cinderx            # source-built dynamic CinderX into the plain venv (configs 2-4)
  install_cinderx_pythonlib  # PythonLib .pth into the static venv (configs 6-8)
  install_cinderx_pip        # pre-built PyPI wheel into the pip venv (configs 9-12; REQUIRED unless --skip-pip)
  gen_jitlists               # generated once with the plain venv; shared by both
  run_builtin
  run_fastmark
  run_static
  run_pyperf
  write_summary
  ok "All done. See $RESULTS/SUMMARY.md"
}

main "$@"
