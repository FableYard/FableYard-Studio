# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

import json
from pathlib import Path
from typing import Any, Dict


class ComponentConfig:
    def __init__(self):
        self._path = None

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        attrs_str = ", ".join(f"{k}={repr(v)}" for k, v in sorted(attrs.items()))

        return f"ComponentConfig({attrs_str})"

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    def load_config(self) -> None:
        """
        Load configuration from the file specified in config_path.

        Supports JSON files. Configuration keys are set as attributes on this instance.

        Raises:
            FileNotFoundError: If config_path is not set or the file doesn't exist
            ValueError: If the config file format is invalid
        """
        if self._path is ...:
            raise ValueError("config_path must be set before loading config")

        file = Path(self._path)
        if not file.exists():
            raise FileNotFoundError(f"Config file not found: {file}")

        with open(file, 'r', encoding='utf-8') as f:
            config_data: Dict[str, Any] = json.load(f)

        for key, value in config_data.items():
            if key.startswith('_'):
                pass
            else:
                setattr(self, key, value)
