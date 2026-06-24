"""
Single VDevice shared between YOLO detector and ReID model.
Uses Hailo's ROUND_ROBIN scheduler so both networks share the chip
concurrently without explicit activate/deactivate per call.
"""
import threading
from hailo_platform import (
    VDevice, HEF, ConfigureParams, HailoStreamInterface,
    HailoSchedulingAlgorithm,
)


class HailoBackend:
    """Singleton-ish: one VDevice with scheduler, multiple network groups."""
    _instance = None
    _lock     = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdev = VDevice(params)
        self._network_groups = {}
        print("[HailoBackend] VDevice created with ROUND_ROBIN scheduler")

    def configure(self, hef_path: str, name: str):
        """Load a HEF onto the shared VDevice. Returns the network group."""
        if name in self._network_groups:
            return self._network_groups[name]
        hef = HEF(hef_path)
        cfg = ConfigureParams.create_from_hef(
            hef=hef, interface=HailoStreamInterface.PCIe)
        ngrp = self.vdev.configure(hef, cfg)[0]
        self._network_groups[name] = (hef, ngrp)
        print(f"[HailoBackend] Loaded '{name}' from {hef_path}")
        return hef, ngrp
