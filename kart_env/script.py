from dolphin import event, gui, controller, savestate
import socket
import struct
import os

instance_id = int(os.environ["INSTANCE_ID"])
num_agents = int(os.environ["NUM_AGENTS"])
port = 9999 + instance_id
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("localhost", port))

GC_BUTTONS = ["A", "B", "X", "Y", "Z", "Start", "Up", "Down", "Left", "Right", "L", "R"]
AGENT_STRUCT = "<H6f"
AGENT_SIZE = struct.calcsize(AGENT_STRUCT)

red = 0xFFFF0000
frame_counter = 0
while True:
    data = sock.recv(1024)
    payload = data[4:]

    if data.startswith(b"step"):
        for i in range(num_agents):
            offset = i * AGENT_SIZE
            btn_mask, sx, sy, csx, csy, tl, tr = struct.unpack_from(
                AGENT_STRUCT, payload, offset
            )

            inputs = {
                btn: bool(btn_mask & (1 << j)) for j, btn in enumerate(GC_BUTTONS)
            }
            inputs.update(
                {
                    "StickX": sx,
                    "StickY": sy,
                    "CStickX": csx,
                    "CStickY": csy,
                    "TriggerLeft": tl,
                    "TriggerRight": tr,
                }
            )
            controller.set_gc_buttons(i, inputs)

        await event.frameadvance()
        sock.sendall(b"step_done")

    elif data.startswith(b"close"):
        sock.close()
        break

    elif data.startswith(b"savefile"):
        payload = data[8:]
        path = payload.decode()
        savestate.save_to_file(path)
        sock.sendall(b"save_done")

    elif data.startswith(b"saveslot"):
        payload = data[8:]
        slot = int(payload.decode())
        savestate.save_to_slot(slot)
        sock.sendall(b"save_done")

    elif data.startswith(b"loadfile"):
        payload = data[8:]
        path = payload.decode()
        savestate.load_from_file(path)
        await event.frameadvance()
        sock.sendall(b"load_done")

    elif data.startswith(b"loadslot"):
        payload = data[8:]
        slot = int(payload.decode())
        savestate.load_from_slot(slot)
        await event.frameadvance()
        sock.sendall(b"load_done")

    frame_counter += 1
    # draw on screen
    gui.draw_text((10, 10), red, f"Frame: {frame_counter}")
