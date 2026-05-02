"""Hatch build hook to embed git hash at build time."""

import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_hash = "unknown"

        build_info_path = Path(self.root) / "agentbridge" / "_build_info.py"
        build_info_path.write_text(f'GIT_HASH = "{git_hash}"\n')

        # Force include the generated file
        build_data["force_include"][str(build_info_path)] = "agentbridge/_build_info.py"
