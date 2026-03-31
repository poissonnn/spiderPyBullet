"""
Command-line interface for AutoPyPack.

This module provides the CLI commands for scanning projects and installing
missing Python packages.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import TYPE_CHECKING

from .core import install_package, is_module_available, load_mappings, scan_imports

if TYPE_CHECKING:
    from collections.abc import Sequence

# Configure module logger
logger = logging.getLogger(__name__)


def find_python_files(directory: str) -> list[str]:
    """
    Recursively find all Python files in the directory.

    Args:
        directory: The root directory to search.

    Returns:
        list[str]: A list of absolute paths to Python files.
    """
    python_files: list[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def collect_all_imports(directory: str, quiet: bool = False) -> set[str]:
    """
    Scan all Python files in the directory and collect imports.

    Args:
        directory: The root directory to scan.
        quiet: If True, suppress informational output.

    Returns:
        set[str]: A set of all unique import names found.
    """
    python_files = find_python_files(directory)
    all_imports: set[str] = set()

    if not quiet:
        logger.info("Scanning %d Python files for imports...", len(python_files))
        print(f"[AutoPyPack] Scanning {len(python_files)} Python files for imports...")

    for file_path in python_files:
        try:
            imports = scan_imports(file_path)
            all_imports.update(imports)
        except Exception as e:
            if not quiet:
                logger.warning("Error scanning %s: %s", file_path, e)
                print(f"[AutoPyPack] Error scanning {file_path}: {e}")

    return all_imports


def is_local_module(module_name: str, directory: str) -> bool:
    """
    Check if a module name refers to a local Python file or directory in the project.

    Args:
        module_name: The name of the module to check.
        directory: The root directory of the project.

    Returns:
        bool: True if the module is a local module, False otherwise.
    """
    # Check if there's a .py file with this name
    py_file = os.path.join(directory, f"{module_name}.py")
    if os.path.isfile(py_file):
        return True

    # Check if there's a directory with this name (might be a package)
    dir_path = os.path.join(directory, module_name)
    if os.path.isdir(dir_path):
        # Check if it has an __init__.py to confirm it's a package
        init_file = os.path.join(dir_path, "__init__.py")
        if os.path.isfile(init_file):
            return True

        # Even without __init__.py, if there are Python files inside, consider it local
        for _, _, files in os.walk(dir_path):
            if any(f.endswith(".py") for f in files):
                return True

    # Recursively check in subdirectories
    for root, dirs, _ in os.walk(directory):
        for d in dirs:
            # Check each subdirectory for module
            subdir_path = os.path.join(root, d)
            py_file = os.path.join(subdir_path, f"{module_name}.py")
            dir_path = os.path.join(subdir_path, module_name)

            if os.path.isfile(py_file):
                return True

            if os.path.isdir(dir_path):
                init_file = os.path.join(dir_path, "__init__.py")
                if os.path.isfile(init_file):
                    return True

                # Check for any Python files inside
                for _, _, files in os.walk(dir_path):
                    if any(f.endswith(".py") for f in files):
                        return True

    return False


# Standard library modules set - more comprehensive and maintainable approach
# This set is used for quick lookup of stdlib modules
STDLIB_MODULES: frozenset[str] = frozenset([
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio", "asyncore",
    "atexit", "audioop", "base64", "bdb", "binascii", "binhex", "bisect",
    "builtins", "bz2", "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd",
    "code", "codecs", "codeop", "collections", "colorsys", "compileall",
    "concurrent", "configparser", "contextlib", "contextvars", "copy", "copyreg",
    "cProfile", "crypt", "csv", "ctypes", "curses", "dataclasses", "datetime",
    "dbm", "decimal", "difflib", "dis", "distutils", "doctest", "email",
    "encodings", "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput",
    "fnmatch", "fractions", "ftplib", "functools", "gc", "getopt", "getpass",
    "gettext", "glob", "graphlib", "grp", "gzip", "hashlib", "heapq", "hmac",
    "html", "http", "idlelib", "imaplib", "imghdr", "imp", "importlib", "inspect",
    "io", "ipaddress", "itertools", "json", "keyword", "lib2to3", "linecache",
    "locale", "logging", "lzma", "mailbox", "mailcap", "marshal", "math",
    "mimetypes", "mmap", "modulefinder", "msvcrt", "multiprocessing", "netrc",
    "nis", "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev",
    "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
    "plistlib", "poplib", "posix", "posixpath", "pprint", "profile", "pstats",
    "pty", "pwd", "py_compile", "pyclbr", "pydoc", "queue", "quopri", "random",
    "re", "readline", "reprlib", "resource", "rlcompleter", "runpy", "sched",
    "secrets", "select", "selectors", "shelve", "shlex", "shutil", "signal",
    "site", "smtpd", "smtplib", "sndhdr", "socket", "socketserver", "spwd",
    "sqlite3", "ssl", "stat", "statistics", "string", "stringprep", "struct",
    "subprocess", "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny",
    "tarfile", "telnetlib", "tempfile", "termios", "test", "textwrap", "threading",
    "time", "timeit", "tkinter", "token", "tokenize", "tomllib", "trace",
    "traceback", "tracemalloc", "tty", "turtle", "turtledemo", "types", "typing",
    "unicodedata", "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave",
    "weakref", "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib", "xml",
    "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib", "zoneinfo",
])


def is_stdlib_module(module_name: str) -> bool:
    """
    Check if a module is part of the Python standard library.

    This function uses multiple strategies to determine if a module is part of
    the standard library:
    1. Check if it's a builtin module
    2. Check against a known set of stdlib module names
    3. Check if the module exists in non-site-packages paths

    Args:
        module_name: The name of the module to check.

    Returns:
        bool: True if the module is part of stdlib, False otherwise.
    """
    # Check builtin modules first (fastest)
    if module_name in sys.builtin_module_names:
        return True

    # Check against our known stdlib set
    if module_name in STDLIB_MODULES:
        return True

    # Check if module is in stdlib paths (not site-packages)
    for path in sys.path:
        if "site-packages" in path or "dist-packages" in path:
            continue

        module_path = os.path.join(path, module_name)
        module_file = os.path.join(path, f"{module_name}.py")

        if os.path.exists(module_path) or os.path.exists(module_file):
            return True

    return False


def install_missing_packages(directory: str, quiet: bool = False) -> None:
    """
    Install missing packages for all imports found in the directory.

    Args:
        directory: The root directory to scan for Python files.
        quiet: If True, suppress informational output.
    """
    mappings = load_mappings()
    all_imports = collect_all_imports(directory, quiet)

    if not all_imports:
        if not quiet:
            logger.info("No imports found in the project.")
            print("[AutoPyPack] No imports found in the project.")
        return

    if not quiet:
        logger.info("Found %d unique imports.", len(all_imports))
        print(f"[AutoPyPack] Found {len(all_imports)} unique imports.")

    # Get project directory name to exclude internal modules
    project_name = os.path.basename(os.path.abspath(directory))
    internal_modules: list[str] = [
        "autopypack", "AutoPyPack", "core", "cli", project_name.lower()
    ]

    # Get the current module name to avoid considering it as a dependency
    current_module_name = os.path.basename(
        os.path.dirname(os.path.abspath(__file__))
    ).lower()
    if current_module_name not in internal_modules:
        internal_modules.append(current_module_name)

    missing_packages: list[tuple[str, str]] = []
    for module_name in all_imports:
        # Skip internal modules and standard library modules
        if (
            module_name.lower() in [m.lower() for m in internal_modules]
            or is_stdlib_module(module_name)
        ):
            continue

        # Skip if it's a local module in the project directory
        if is_local_module(module_name, directory):
            if not quiet:
                logger.debug("Detected local module: %s", module_name)
                print(f"[AutoPyPack] Detected local module: {module_name}")
            continue

        # Skip if module is already available
        if not is_module_available(module_name):
            package_name = mappings.get(module_name, module_name)
            missing_packages.append((module_name, package_name))

    if not missing_packages:
        if not quiet:
            logger.info("All packages are already installed!")
            print("[AutoPyPack] All packages are already installed! ✅")
        return

    if not quiet:
        logger.info("Found %d missing packages to install.", len(missing_packages))
        print(f"[AutoPyPack] Found {len(missing_packages)} missing packages to install.")

    for module_name, package_name in missing_packages:
        install_package(package_name, quiet)

    if not quiet:
        logger.info("Package installation complete!")
        print("[AutoPyPack] Package installation complete! ✅")


def list_project_modules(directory: str, quiet: bool = False) -> list[str]:
    """
    List all non-standard library modules used in the project.

    Args:
        directory: The root directory to scan.
        quiet: If True, only output package names without additional info.

    Returns:
        list[str]: A sorted list of external package names.
    """
    mappings = load_mappings()
    all_imports = collect_all_imports(directory, quiet)

    if not all_imports and not quiet:
        logger.info("No imports found in the project.")
        print("[AutoPyPack] No imports found in the project.")
        return []

    if not quiet:
        logger.info("Found %d unique imports.", len(all_imports))
        print(f"[AutoPyPack] Found {len(all_imports)} unique imports.")

    # Get project directory name to exclude internal modules
    project_name = os.path.basename(os.path.abspath(directory))
    internal_modules: list[str] = [
        "autopypack", "AutoPyPack", "core", "cli", project_name.lower()
    ]

    # Get the current module name to avoid considering it as a dependency
    current_module_name = os.path.basename(
        os.path.dirname(os.path.abspath(__file__))
    ).lower()
    if current_module_name not in internal_modules:
        internal_modules.append(current_module_name)

    external_packages: list[str] = []
    for module_name in all_imports:
        # Skip internal modules and standard library modules
        if (
            module_name.lower() in [m.lower() for m in internal_modules]
            or is_stdlib_module(module_name)
        ):
            continue

        # Skip if it's a local module in the project directory
        if is_local_module(module_name, directory):
            if not quiet:
                logger.debug("Detected local module: %s", module_name)
                print(f"[AutoPyPack] Detected local module: {module_name}")
            continue

        package_name = mappings.get(module_name, module_name)
        if package_name not in external_packages:
            external_packages.append(package_name)

    external_packages.sort()  # Sort alphabetically for readability

    if not quiet:
        if not external_packages:
            logger.info("No external packages found.")
            print("[AutoPyPack] No external packages found.")
        else:
            logger.info("Found %d external packages.", len(external_packages))
            print(f"[AutoPyPack] Found {len(external_packages)} external packages:")
            for package in external_packages:
                print(package)

    return external_packages


def main(args: Sequence[str] | None = None) -> None:
    """
    Main entry point for the CLI.

    Args:
        args: Command-line arguments. If None, uses sys.argv.
    """
    parser = argparse.ArgumentParser(
        description="AutoPyPack - Automatically install missing Python packages",
        prog="autopypack",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Install command
    install_parser = subparsers.add_parser(
        "install",
        aliases=["i"],
        help="Scan project and install missing packages",
    )
    install_parser.add_argument(
        "--dir", "-d",
        default=".",
        help="Directory to scan (default: current directory)",
    )
    install_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress informational output",
    )

    # List command
    list_parser = subparsers.add_parser(
        "list",
        aliases=["l"],
        help="List all external packages used in the project",
    )
    list_parser.add_argument(
        "--dir", "-d",
        default=".",
        help="Directory to scan (default: current directory)",
    )
    list_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only output package names without additional info (good for requirements.txt)",
    )

    # Parse arguments
    parsed_args = parser.parse_args(args)

    if parsed_args.command in ["install", "i"]:
        directory = os.path.abspath(parsed_args.dir)
        if not parsed_args.quiet:
            print(f"[AutoPyPack] Scanning directory: {directory}")
        install_missing_packages(directory, parsed_args.quiet)
    elif parsed_args.command in ["list", "l"]:
        directory = os.path.abspath(parsed_args.dir)
        if not parsed_args.quiet:
            print(f"[AutoPyPack] Scanning directory: {directory}")
        packages = list_project_modules(directory, parsed_args.quiet)
        if parsed_args.quiet and packages:
            # Print only package names, one per line (good for redirection to requirements.txt)
            for package in packages:
                print(package)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 