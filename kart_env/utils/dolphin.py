import pathlib
from typing import Any, Dict
import subprocess
import os
import shutil
import time
import socket
import struct

from .macro_helper import OptionType
from .enums import (
    VNC_BASE,
    NOVNC_BASE,
    SOCK_BASE,
    DOLPHIN_PATH,
    GC_BUTTONS,
    NEUTRAL_ACTION,
)

ObsType = Dict[str, Any]
ActionType = Dict[str, Any]
AgentID = int


class Dolphin:
    def __init__(self, instance_id: int, options: OptionType | None = None):
        self.options = options if options is not None else OptionType()
        self._closed = False

        self.possible_agents = [i for i in range(self.options.num_agents)]
        self.agents = self.possible_agents

        self.instance_id = instance_id
        self.vnc_port = VNC_BASE + instance_id
        self.novnc_port = NOVNC_BASE + instance_id
        self.sock_port = SOCK_BASE + instance_id
        self.user_dir = pathlib.Path("users") / str(instance_id)
        self.dolphin_proc_pid: int | None = None

        if not self.user_dir.exists():
            dolphin_settings_path = pathlib.Path(__file__).parent.parent / "dolphin_settings"
            shutil.copytree(dolphin_settings_path, self.user_dir)
            self.options.is_license_created = False

        self.processes: list[subprocess.Popen] = []
        self._run_env()

    def click(self, actions: dict[AgentID, ActionType], num_frame: int = 250):
        self._send_actions(actions)
        for _ in range(num_frame):
            self._send_actions({})

    def async_save_file(self, path: str):
        self.conn.sendall(b"savefile" + path.encode())

    def async_save_slot(self, slot: int):
        self.conn.sendall(b"saveslot" + str(slot).encode())

    def await_save_done(self):
        assert self.conn.recv(1024) == b"save_done"

    def save_file(self, path: str):
        self.async_save_file(path)
        self.await_save_done()

    def save_slot(self, slot: int):
        self.async_save_slot(slot)
        self.await_save_done()

    def async_load_file(self, path: str):
        self.conn.sendall(b"loadfile" + path.encode())

    def async_load_slot(self, slot: int):
        self.conn.sendall(b"loadslot" + str(slot).encode())

    def await_load_done(self):
        assert self.conn.recv(1024) == b"load_done"

    def load_file(self, path: str):
        self.async_load_file(path)
        self.await_load_done()

    def load_slot(self, slot: int):
        self.async_load_slot(slot)
        self.await_load_done()

    def _run_env(self):
        stdout = subprocess.DEVNULL if not self.options.verbose else None
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.server_sock.bind(("localhost", self.sock_port))
        self.server_sock.listen(1)

        xvfb_command = ["Xvfb", f":{self.instance_id}", "-screen", "0", "640x480x24"]
        self.processes.append(subprocess.Popen(xvfb_command, stdout=stdout))
        while not os.path.exists(f"/tmp/.X11-unix/X{self.instance_id}"):
            time.sleep(0.1)

        x11vnc_command = [
            "x11vnc",
            "-display",
            f":{self.instance_id}",
            "-forever",
            "-nopw",
            "-shared",
            "-rfbport",
            str(self.vnc_port),
        ]
        self.processes.append(subprocess.Popen(x11vnc_command, stdout=stdout))

        websockify_command = [
            "websockify",
            "--web",
            "/usr/share/novnc/",
            str(self.novnc_port),
            f"localhost:{self.vnc_port}",
        ]
        self.processes.append(subprocess.Popen(websockify_command, stdout=stdout))

        script_path = pathlib.Path(__file__).parent.parent / "script.py"
        dolphin_command = [
            "vglrun",
            "-d",
            "egl0",
            DOLPHIN_PATH,
            "--batch",
            f"--user={self.user_dir}",
            "-e",
            "MarioKartWii.iso",
            "--script",
            script_path,
        ]
        dolphin_env = os.environ.copy()
        dolphin_env["DISPLAY"] = f":{self.instance_id}"
        dolphin_env["INSTANCE_ID"] = f"{self.instance_id}"
        dolphin_env["NUM_AGENTS"] = f"{self.options.num_agents}"
        dolphin_process = subprocess.Popen(
            dolphin_command, env=dolphin_env, stdout=stdout
        )
        self.processes.append(dolphin_process)

        self.dolphin_proc_pid = dolphin_process.pid
        self.conn, _ = self.server_sock.accept()

    def _pack_actions(self, actions: dict[AgentID, ActionType]) -> bytes:
        data = []
        for i in range(len(self.agents)):
            action = NEUTRAL_ACTION.copy()
            if i in actions:
                action.update(actions[i])

            button_mask = 0
            for j, btn in enumerate(GC_BUTTONS):
                if action[btn]:
                    button_mask |= 1 << j

            data.append(button_mask)
            data.extend(
                [
                    action["StickX"],
                    action["StickY"],
                    action["CStickX"],
                    action["CStickY"],
                    action["TriggerLeft"],
                    action["TriggerRight"],
                ]
            )

        return struct.pack("<" + "H6f" * self.options.num_agents, *data)

    def async_send_actions(self, actions: dict[AgentID, ActionType]):
        payload = b"step" + self._pack_actions(actions)
        self.conn.sendall(payload)

    def await_send_actions(self):
        assert self.conn.recv(1024) == b"step_done"

    def _send_actions(self, actions: dict[AgentID, ActionType]):
        self.async_send_actions(actions)
        self.await_send_actions()

    def close(self):
        if self._closed:
            return
        self._closed = True

        self.conn.sendall(b"close")
        self.conn.close()
        self.server_sock.close()

        try:
            for proc in reversed(self.processes):
                proc.terminate()
                proc.wait()

        except Exception:
            pass
