import asyncio
import json
import queue
from dataclasses import asdict
from threading import Thread
from time import sleep, time
from typing import Callable

from websockets.asyncio.client import connect

from xiron_py.data import LaserScan, Pose, Twist
from xiron_py.protos.any_pb2 import Any
from xiron_py.protos.laser_scan_pb2 import LaserScanMsg
from xiron_py.protos.pose_pb2 import PoseMsg
from xiron_py.protos.reset_pb2 import ResetMsg
from xiron_py.protos.twist_pb2 import TwistMsg


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

    async def _send_scan_message(self, msg):
        decoded_msg = LaserScanMsg()
        decoded_msg.ParseFromString(msg)
        msg = LaserScan(
            timestamp=decoded_msg.timestamp,
            robot_id=decoded_msg.robot_id,
            angle_min=decoded_msg.angle_min,
            angle_max=decoded_msg.angle_max,
            angle_step=decoded_msg.angle_step,
            num_readings=decoded_msg.num_readings,
            values=decoded_msg.values,
        )
        await asyncio.to_thread(self._scan_callbacks[msg.robot_id], msg)

    async def _send_pose_message(self, msg):
        decoded_msg = PoseMsg()
        decoded_msg.ParseFromString(msg.value)

        msg = Pose(
            timestamp=decoded_msg.timestamp,
            robot_id=decoded_msg.robot_id,
            position=[decoded_msg.position.x, decoded_msg.position.y],
            orientation=decoded_msg.orientation,
        )
        await asyncio.to_thread(self._pose_callbacks[msg.robot_id], msg)


    async def reciever_coroutine(self):
        try:
            async with connect("ws://localhost:9001", ping_interval=None) as websocket:
                while True:
                    message = await websocket.recv()
                    wrapped_msg = Any()
                    wrapped_msg.ParseFromString(message)
                    
                    print(wrapped_msg)

                    if wrapped_msg.type_url == "scan":
                        await self._send_scan_message(wrapped_msg.value)
                        
                    elif wrapped_msg.type_url == "pose":
                        await self._send_pose_message(wrapped_msg.value)
                        
                    else:
                        print("WARNING: unknown message type")
        except Exception as e:
            print(f"Exception in Reciever Coroutine: {e}")

    async def sender_coroutine(self):
        try:
            async with connect("ws://localhost:9000", ping_interval=None) as websocket:
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
        message = ResetMsg()
        message.timestamp = self.now()

        wrapped_msg = Any()
        wrapped_msg.type_url = "reset"
        wrapped_msg.value = message.SerialiseToString()

        self.data_to_send.put(wrapped_msg.SerialiseToString())

    def publish_velocity(self, msg: Twist):
        message = TwistMsg()
        message.timestamp = msg.timestamp
        message.robot_id = msg.robot_id
        message.linear.x = msg.linear[0]
        message.linear.y = msg.linear[1]
        message.angular = msg.angular

        wrapped_msg = Any()
        wrapped_msg.type_url = "reset"
        wrapped_msg.value = message.SerialiseToString()

        self.data_to_send.put(wrapped_msg.SerialiseToString())

    def now(self):
        return time()
