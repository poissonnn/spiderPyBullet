"""
Core functionality for AutoPyPack.

This module provides the main functions for scanning Python files for imports,
checking module availability, and installing packages.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import os
import subprocess
import sys
from typing import Any

# Configure module logger
logger = logging.getLogger(__name__)


def load_mappings() -> dict[str, str]:
    """
    Load the import-to-package name mappings from mappings.json.

    Returns:
        dict[str, str]: A dictionary mapping import names to PyPI package names.
    """
    path = os.path.join(os.path.dirname(__file__), "mappings.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            mappings: dict[str, str] = json.load(f)
            return mappings
    except FileNotFoundError:
        logger.warning("Could not find mappings.json at %s", path)
        return {}
    except json.JSONDecodeError as e:
        logger.error("Failed to parse mappings.json: %s", e)
        return {}


install_name_mapping: dict[str, str] = load_mappings()


def install_package(package_name: str, quiet: bool = False) -> bool:
    """
    Install a Python package using pip.

    Args:
        package_name: The name of the package to install (PyPI name).
        quiet: If True, suppress installation output.

    Returns:
        bool: True if installation succeeded, False otherwise.
    """
    try:
        if not quiet:
            logger.info("Installing: %s ...", package_name)
            print(f"[AutoPyPack] Installing: {package_name} ...")

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name],
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=subprocess.DEVNULL if quiet else None,
        )
        return True
    except subprocess.CalledProcessError:
        if not quiet:
            logger.error("Failed to install %s. Please install manually.", package_name)
            print(f"[AutoPyPack] ❌ Failed to install {package_name}. Please install manually.")
        return False


def is_module_available(module_name: str) -> bool:
    """
    Check if a module is available in the current Python environment.

    Args:
        module_name: The name of the module to check.

    Returns:
        bool: True if the module is available, False otherwise.
    """
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


def scan_imports(file_path: str) -> set[str]:
    """
    Scan a Python file for import statements and return a set of imported module names.

    This function parses the Python file using the AST module and extracts
    all top-level module names from import and import-from statements.

    Args:
        file_path: Path to the Python file to scan.

    Returns:
        set[str]: Set of top-level imported module names.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except OSError as e:
        logger.error("Error reading %s: %s", file_path, e)
        print(f"[AutoPyPack] Error reading {file_path}: {e}")
        return set()

    try:
        node = ast.parse(content, filename=file_path)
    except SyntaxError as e:
        logger.error("Syntax error in %s: %s", file_path, e)
        print(f"[AutoPyPack] Syntax error in {file_path}: {e}")
        return set()

    imported_modules: set[str] = set()

    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                # Only add the top-level module name
                imported_modules.add(alias.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                # Only add the top-level module name from import-from statements
                imported_modules.add(n.module.split(".")[0])

    logger.debug("Found %d imports in %s", len(imported_modules), file_path)
    return imported_modules


def main() -> None:
    """
    Legacy entry point function for backward compatibility.

    This function is kept for backward compatibility with older usage patterns.
    For new code, use the CLI interface instead.
    """
    if len(sys.argv) < 2:
        print("[AutoPyPack] Error: No file specified.")
        print("[AutoPyPack] Usage: python -m autopypack <file.py>")
        return

    script_path = sys.argv[1]
    if not os.path.exists(script_path):
        print(f"[AutoPyPack] Error: File not found: {script_path}")
        return

    print(f"[AutoPyPack] Scanning file: {script_path}")
    modules = scan_imports(script_path)

    if not modules:
        print("[AutoPyPack] No imports found.")
        return

    print(f"[AutoPyPack] Found {len(modules)} imports.")

    for mod in modules:
        if not is_module_available(mod):
            install_name = install_name_mapping.get(mod, mod)
            install_package(install_name)

    print("[AutoPyPack] Done!") 