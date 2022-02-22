""" doit tasks for ipyml

    Generally, you'll just want to `doit`.

    See `doit list` for more options.
"""

import os
import shutil
import subprocess
from hashlib import sha256

from doit.action import CmdAction
from doit.tools import LongRunning, PythonInteractiveAction, config_changed
from yaml import safe_load

import _scripts.project as P
import _scripts.utils as U

os.environ.update(
    CONDA_EXE="mamba",
    CONDARC=str(P.CONDARC),
    MAMBA_NO_BANNER="1",
    PIP_DISABLE_PIP_VERSION_CHECK="1",
    PYTHONIOENCODING="utf-8",
)

DOIT_CONFIG = {
    "backend": "sqlite3",
    "verbosity": 2,
    "par_type": "thread",
}

COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8")

PROJ = safe_load(P.PROJ.read_text(encoding="utf-8"))


def task_preflight():
    """ensure a sane development environment"""
    file_dep = [P.PROJ, P.SCRIPTS / "preflight.py"]  # P.PROJ_LOCK inistead of P.PROJ GH#3

    yield _ok(
        dict(
            uptodate=[config_changed({"commit": COMMIT})],
            name="conda",
            file_dep=file_dep,
            actions=(
                [_echo_ok("skipping preflight, hope you know what you're doing!")]
                if P.SKIP_CONDA_PREFLIGHT
                else [[*P.PREFLIGHT, "conda"]]
            ),
        ),
        P.OK_PREFLIGHT_CONDA,
    )

    yield _ok(
        dict(
            name="lab",
            file_dep=[*file_dep, P.OK_ENV["develop"]],
            actions=[[*P.APR_DEV, *P.PREFLIGHT, "lab"]],
        ),
        P.OK_PREFLIGHT_LAB,
    )


def task_env():
    """prepare project envs"""
    envs = ["develop",]
    for i, env in enumerate(envs):
        file_dep = [P.PROJ, P.OK_PREFLIGHT_CONDA]  # P.PROJ_LOCK instead of P.PROJ GH#3
        if P.FORCE_SERIAL_ENV_PREP and i:
            file_dep += [P.OK_ENV[envs[i - 1]]]
        yield _ok(
            dict(name=env, file_dep=file_dep, actions=[[*P.AP_PREP, env]]),
            P.OK_ENV[env],
        )


# def task_release():
#     """everything we'd need to do to release (except release)"""
#     return _ok(
#         dict(
#             file_dep=[
#                 *P.EXAMPLE_HTML,
#                 P.OK_LINT,
#                 P.OK_PIP_INSTALL,
#                 P.OK_PREFLIGHT_RELEASE,
#                 P.SHA256SUMS,
#                 P.HTML_COV_INDEX,
#             ],
#             actions=[_echo_ok("ready to release")],
#         ),
#         P.OK_RELEASE,
#     )


def task_setup():
    """perform all setup activities"""

    _install = ["--no-deps", "--ignore-installed", "-vv"]

    # if P.INSTALL_ARTIFACT == "wheel":
    #     _install += [P.WHEEL]
    # elif P.INSTALL_ARTIFACT == "sdist":
    #     _install += [P.SDIST]
    # else:
    _install += ["-e", "."]

    yield _ok(
        dict(
            name="py",
            file_dep=[P.SETUP_PY, P.SETUP_CFG, P.OK_ENV["develop"]],  # P.WHEEL, P.SDIST once release is added
            uptodate=[config_changed({"artifact": P.INSTALL_ARTIFACT})],
            actions=[
                [*P.APR_DEV, *P.PIP, "install", *_install],
                [*P.APR_DEV, *P.PIP, "check"],
            ],
        ),
        P.OK_PIP_INSTALL,
    )


# def task_build():
#     """build packages"""
#     yield dict(
#         name="py",
#         file_dep=[
#             *P.ALL_PY_SRC,
#             P.SETUP_CFG,
#             P.SETUP_PY,
#             P.OK_BLACK,
#             P.OK_ENV["build"],
#         ],
#         actions=[
#             [*P.APR_BUILD, *P.PY, "setup.py", "sdist"],
#             [*P.APR_BUILD, *P.PY, "setup.py", "bdist_wheel"],
#         ],
#         targets=[P.WHEEL, P.SDIST],
#     )

#     def _run_hash():
#         # mimic sha256sum CLI
#         if P.SHA256SUMS.exists():
#             P.SHA256SUMS.unlink()

#         lines = []

#         for p in P.HASH_DEPS:
#             lines += ["  ".join([sha256(p.read_bytes()).hexdigest(), p.name])]

#         output = "\n".join(lines)
#         print(output)
#         P.SHA256SUMS.write_text(output)

#     yield dict(
#         name="hash",
#         file_dep=P.HASH_DEPS,
#         targets=[P.SHA256SUMS],
#         actions=[_run_hash],
#     )


def task_test():
    """run notebook and unit tests"""
    test_deps = [
        P.OK_PYFLAKES,
        P.OK_ENV["develop"],
        P.OK_PIP_INSTALL,
        P.OK_PREFLIGHT_KERNEL,
        *P.ALL_PY_SRC,
    ]

    # def _nb_test(nb):
    #     def _test():
    #         env = dict(os.environ)
    #         env.update(IPYML_TESTING="true")
    #         args = [
    #             *P.APR_DEV,
    #             "jupyter",
    #             "nbconvert",
    #             "--to",
    #             "html",
    #             "--output-dir",
    #             P.BUILD_NBHTML,
    #             "--execute",
    #             "--ExecutePreprocessor.timeout=1200",
    #             nb,
    #         ]
    #         return CmdAction(args, env=env, shell=False)

    #     return dict(
    #         name=f"nb:{nb.name}".replace(" ", "_").replace(".ipynb", ""),
    #         file_dep=[*P.EXAMPLE_IPYNB, *test_deps],
    #         actions=[_test()],
    #         targets=[P.BUILD_NBHTML / nb.name.replace(".ipynb", ".html")],
    #     )

    # for nb in P.EXAMPLE_IPYNB:
    #     yield _nb_test(nb)

    yield dict(
        name="pytest",
        file_dep=[*test_deps],
        actions=[
            lambda: shutil.rmtree(P.HTML_COV_INDEX.parent)
            if P.HTML_COV_INDEX.parent.exists()
            else None,
            [
                *P.APR_DEV,
                "pytest",
                "-vv",
                "--ff",
                "--cov",
                "ipyml",
                "--cov-report",
                "html:build/htmlcov",
                "--cov-report",
                "term-missing:skip-covered",
                "--no-cov-on-fail",
                P.PY_TESTS,
            ],
            lambda: U.strip_timestamps(P.HTML_COV_INDEX),
        ],
        targets=[P.HTML_COV_INDEX],
    )


# def task_lint():
#     """format all source files"""

#     yield _ok(
#         dict(
#             name="black",
#             file_dep=[*P.ALL_PY, P.OK_ENV["qa"]],
#             actions=[
#                 [*P.APR_QA, "isort", *P.ALL_PY],
#                 [*P.APR_QA, "black", "--quiet", *P.ALL_PY],
#             ],
#         ),
#         P.OK_BLACK,
#     )
#     yield _ok(
#         dict(
#             name="flake8",
#             file_dep=[*P.ALL_PY, P.OK_BLACK],
#             actions=[[*P.APR_QA, "flake8", *P.ALL_PY]],
#         ),
#         P.OK_FLAKE8,
#     )
#     yield _ok(
#         dict(
#             name="pyflakes",
#             file_dep=[*P.ALL_PY, P.OK_BLACK],
#             actions=[[*P.APR_QA, "pyflakes", *P.ALL_PY]],
#         ),
#         P.OK_PYFLAKES,
#     )
#     yield _ok(
#         dict(
#             name="prettier",
#             file_dep=[P.YARN_INTEGRITY, *P.ALL_PRETTIER, P.OK_ENV["qa"]],
#             actions=[[*P.APR_QA, *P.JLPM, "--silent", "lint"]],
#         ),
#         P.OK_PRETTIER,
#     )

#     def _nblint(nb, nb_ok):
#         return _ok(
#             dict(
#                 name=f"""nblint:{nb.stem}""",
#                 file_dep=[P.YARN_INTEGRITY, P.OK_ENV["qa"], nb],
#                 actions=[
#                     LongRunning([*P.APR_QA, *P.PYM, "_scripts.nblint", nb], shell=False)
#                 ],
#             ),
#             nb_ok,
#         )

#     all_nb_lint_ok = {nb: P.OK_NBLINT / nb.stem for nb in P.EXAMPLE_IPYNB}

#     for nb, nb_ok in all_nb_lint_ok.items():
#         yield _nblint(nb, nb_ok)

#     yield _ok(
#         dict(
#             name="all",
#             actions=[_echo_ok("all ok")],
#             file_dep=[
#                 P.OK_BLACK,
#                 P.OK_FLAKE8,
#                 P.OK_PRETTIER,
#                 P.OK_PYFLAKES,
#                 *all_nb_lint_ok.values(),
#             ],
#         ),
#         P.OK_LINT,
#     )


def task_lab():
    """run JupyterLab "normally" (not watching sources)
    
    TODO add lab env spec
    """

    def lab():
        proc = subprocess.Popen(
            list(map(str, [*P.APR_DEV,])), stdin=subprocess.PIPE
        )

        try:
            proc.wait()
        except KeyboardInterrupt:
            print("attempting to stop lab, you may want to check your process monitor")
            proc.terminate()
            proc.communicate(b"y\n")

        proc.wait()
        return True

    return dict(
        uptodate=[lambda: False],
        file_dep=[P.OK_PIP_INSTALL, P.OK_PREFLIGHT_LAB],
        actions=[PythonInteractiveAction(lab)],
    )


def task_all():
    """do everything except start lab"""
    return dict(
        file_dep=[P.OK_RELEASE, P.OK_PREFLIGHT_LAB],
        actions=([_echo_ok("ALL GOOD")]),
    )


def _echo_ok(msg):
    def _echo():
        print(msg, flush=True)
        return True

    return _echo


def _ok(task, ok):
    task.setdefault("targets", []).append(ok)
    task["actions"] = [
        lambda: [ok.exists() and ok.unlink(), True][-1],
        *task["actions"],
        lambda: [ok.parent.mkdir(exist_ok=True), ok.write_text("ok"), True][-1],
    ]
    return task


def _call(args, **kwargs):
    if "cwd" in kwargs:
        kwargs["cwd"] = str(kwargs["cwd"])
    if "env" in kwargs:
        kwargs["env"] = {k: str(v) for k, v in kwargs["env"].items()}
    args = list(map(str, args))
    print("\n>>>", " ".join(args), "\n", flush=True)
    return subprocess.call(args, **kwargs)


def _channel_args(env="ipyml"):
    return sum([["-c", c] for c in PROJ["env_specs"][env]["channels"]], [])