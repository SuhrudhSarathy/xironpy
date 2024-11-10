import asyncio
import json
import queue
from dataclasses import asdict
from threading import Thread
from time import sleep, time
from typing import Callable

from websockets.asyncio.client import connect

from xiron_py.data import LaserScan, Pose, Twist


class XironContext:
    def __init__(self):
        print("Initialised the main communicator")
        self.data_to_send = queue.Queue()

        # Callbacks
        self._pose_callbacks: dict[str, Callable[[Pose], None]] = {}
        self._scan_callbacks: dict[str, Callable[[Pose], None]] = {}

        self._timer_threads_callbacks = []

    def run_in_separate_thread(self):
        self.run_thread = Thread(target=self.run, daemon=True, name="async_run_thread")
        self.run_thread.start()

    async def main(self):
        await asyncio.gather(
            self.reciever_coroutine(),
            self.sender_coroutine(),
            *self._timer_threads_callbacks,
        )

    def run(self):
        try:
            asyncio.run(self.main())
        except KeyboardInterrupt:
            print("Quitting")

    async def reciever_coroutine(self):
        try:
            async with connect("ws://localhost:9001") as websocket:
                while True:
                    message = await websocket.recv()
                    jsonified_message = json.loads(message)
                    if jsonified_message["type"] == "scan":
                        msg = LaserScan(**jsonified_message["message"])
                        await asyncio.to_thread(self._scan_callbacks[msg.robot_id], msg)
                    elif jsonified_message["type"] == "pose":
                        msg = Pose(**jsonified_message["message"])
                        await asyncio.to_thread(self._pose_callbacks[msg.robot_id], msg)
                    else:
                        print("WARNING: unknown message type")
        except Exception as e:
            print(f"Exception in Reciever Coroutine: {e}")

    async def sender_coroutine(self):
        try:
            async with connect("ws://localhost:9000") as websocket:
                while True:
                    await asyncio.sleep((1 / 50.0))
                    try:
                        data = self.data_to_send.get(block=False)
                        await websocket.send(data)
                    except queue.Empty:
                        pass
        except Exception as e:
            print(f"Exception in Sender Coroutine: {e}")

    def create_pose_subscriber(
        self, robot_id: str, pose_callback: Callable[[Pose], None]
    ):
        self._pose_callbacks[robot_id] = pose_callback

    def create_scan_subscriber(
        self, robot_id: str, scan_callback: Callable[[LaserScan], None]
    ):
        self._scan_callbacks[robot_id] = scan_callback

    def create_timer(self, frequency: float, target_fn: Callable):
        async def callback_fn():
            while True:
                await asyncio.to_thread(target_fn)
                await asyncio.sleep(1 / frequency)

        self._timer_threads_callbacks.append(callback_fn())

    def reset_simulation(self):
        message = {"type": "reset", "message": ""}

        self.data_to_send.put(json.dumps(message))

    def publish_velocity(self, msg: Twist):
        message = {
            "type": "vel",
            "message": asdict(msg),
        }

        self.data_to_send.put(json.dumps(message))

    def now(self):
        return time()
