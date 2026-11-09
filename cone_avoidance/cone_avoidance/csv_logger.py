"""Minimal CSV logger: header comes from the first row's keys."""

from __future__ import annotations

import csv
import os
import time


class CsvLogger:
    def __init__(self, directory: str, name: str, enabled: bool = True):
        self._writer = None
        self._handle = None
        self._fields = None
        if not enabled:
            return
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(directory, f"{name}_{stamp}.csv")
        self._handle = open(self.path, "w", newline="", buffering=1)

    def write(self, row: dict) -> None:
        if self._handle is None:
            return
        if self._writer is None:
            self._fields = list(row.keys())
            self._writer = csv.DictWriter(self._handle, fieldnames=self._fields, extrasaction="ignore")
            self._writer.writeheader()
        self._writer.writerow(row)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
