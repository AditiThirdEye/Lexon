# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "LEXON"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".c", ".cpp"]
if config.lexon_mlir_enable_python_packages:
    config.suffixes.append(".py")

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.lexon_obj_root, "tests")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
    "Examples",
    "Models",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.lexon_obj_root, "tests")
# config.lexon_tools_dir = os.path.join(config.lexon_obj_root, 'bin')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.lexon_tools_dir, config.llvm_tools_dir]
tools = [
    "lexon-opt",
    "lexon-translate",
    "lexon-container-test",
    "lexon-audio-container-test",
    "lexon-text-container-test",
    "mlir-runner",
    "lexon-lenet-run-test-cpu",
]
tools.extend(
    [
        ToolSubst(
            "%mlir_runner_utils_dir",
            config.mlir_runner_utils_dir,
            unresolved="ignore",
        ),
    ]
)

python_executable = config.python_executable
tools.extend(
    [
        ToolSubst("%PYTHON", python_executable, unresolved="ignore"),
    ]
)

# Add the python path for both upstream MLIR and Lexon Compiler python packages.
if config.lexon_mlir_enable_python_packages:
    llvm_config.with_environment(
        "PYTHONPATH",
        [
            os.path.join(
                config.llvm_build_dir,
                "tools",
                "mlir",
                "python_packages",
                "mlir_core",
            ),
            config.lexon_python_packages_dir,
        ],
        append_path=True,
    )

if config.lexon_enable_opencv == "ON":
    tools.append("lexon-image-container-test")

if config.lexon_mlir_enable_dip_lib == "ON":
    tools.append("lexon-new-image-container-test-bmp")
    if config.lexon_enable_png == "ON":
        tools.append("lexon-new-image-container-test-png")

llvm_config.add_tool_substitutions(tools, tool_dirs)
