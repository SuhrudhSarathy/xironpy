import asyncio
import json
import queue
from dataclasses import asdict
from threading import Thread
from time import sleep, time
from typing import Callable

import signal

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from xiron_py.data import LaserScan, Pose, Twist
from xiron_py.protos.any_pb2 import Any
from xiron_py.protos.laser_scan_pb2 import LaserScanMsg
from xiron_py.protos.pose_pb2 import PoseMsg
from xiron_py.protos.reset_pb2 import ResetMsg
from xiron_py.protos.twist_pb2 import TwistMsg


class XironContext:
    def __init__(self, url: str = "localhost", s2c_port: int = 9000, c2s_port: int  = 9001):
        print("Initialised the main communicator")

        self.s2c_url = f"ws://{url}:{s2c_port}"
        self.c2s_url = f"ws://{url}:{c2s_port}"

        self.data_to_send = queue.Queue()

        # Callbacks
        self._pose_callbacks: dict[str, Callable[[Pose], None]] = {}
        self._scan_callbacks: dict[str, Callable[[LaserScan], None]] = {}

        self._timer_threads_callbacks = []

        self._last_recieved_values = {"pose": None, "scan": None}
        self.stop_event = asyncio.Event()

    def run_in_separate_thread(self):
        self.run_thread = Thread(target=self.run, daemon=True, name="async_run_thread")
        self.run_thread.start()

    async def main(self):
        s2c_client_task = asyncio.create_task(self.ws_s2c_client())
        c2s_client_task = asyncio.create_task(self.ws_c2s_client())
        tasks = [asyncio.create_task(t) for t in self._timer_threads_callbacks]

        all_tasks = [s2c_client_task, c2s_client_task, *tasks]

        try:
            await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            print("Main coroutine was cancelled")
        finally:
            print("Cleaning up...")
            for task in all_tasks:
                task.cancel()
            await asyncio.sleep(0.1)  # Allow time for cancellation


    def run(self):
        self.stop_event.clear()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def stop_loop():
            print("Stopping event loop...")
            self.stop_event.set()  # Signal shutdown to coroutines

        loop.add_signal_handler(signal.SIGINT, stop_loop)
        loop.add_signal_handler(signal.SIGTERM, stop_loop)

        try:
            loop.run_until_complete(self.main())  # Pass stop event
        finally:
            print("Shutting down tasks...")
            for task in asyncio.all_tasks(loop):
                task.cancel()
            loop.run_until_complete(asyncio.sleep(0.1))  # Allow cleanup time
            loop.close()
            print("Event loop closed cleanly")


    async def _send_scan_message(self, msg):
        decoded_msg = LaserScanMsg()
        decoded_msg.ParseFromString(msg)
        msg = LaserScan(
            timestamp=decoded_msg.timestamp,
            robot_id=decoded_msg.robot_id,
            angle_min=decoded_msg.angle_min,
            angle_max=decoded_msg.angle_max,
            num_readings=decoded_msg.num_readings,
            values=decoded_msg.values,
        )
        scan_cb =  self._scan_callbacks.get(msg.robot_id)
        if scan_cb is not None:
            await asyncio.to_thread(self._scan_callbacks[msg.robot_id], msg)

    async def _send_pose_message(self, msg):
        decoded_msg = PoseMsg()
        decoded_msg.ParseFromString(msg)

        msg = Pose(
            timestamp=decoded_msg.timestamp,
            robot_id=decoded_msg.robot_id,
            position=[decoded_msg.position.x, decoded_msg.position.y],
            orientation=decoded_msg.orientation,
        )
        pose_cb = self._pose_callbacks.get(msg.robot_id)
        if pose_cb is not None:
            await asyncio.to_thread(self._pose_callbacks[msg.robot_id], msg)

    async def ws_s2c_client(self):
        """ WebSocket client for receiving messages from server to client. """
        try:
            async with connect(self.s2c_url) as websocket:
                print("Connected to S2C WebSocket")

                while not self.stop_event.is_set():
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1)
                        wrapped_msg = Any()
                        wrapped_msg.ParseFromString(message)

                        if wrapped_msg.type_url == "scan":
                            await self._send_scan_message(wrapped_msg.value)
                        elif wrapped_msg.type_url == "pose":
                            await self._send_pose_message(wrapped_msg.value)
                        else:
                            print("WARNING: unknown message type")
                    
                    except asyncio.TimeoutError:
                        continue  # Just retry after timeout
                    except ConnectionClosed as e:
                        print(f"Connection closed: {e}")
                        break
                    except Exception as e:
                        print(f"Error receiving WebSocket message: {e}")

        except Exception as e:
            print(f"Exception in S2C WebSocket client: {e}")

    async def ws_c2s_client(self):
        """ WebSocket client for sending messages from client to server. """
        try:
            async with connect(self.c2s_url) as websocket:
                print("Connected to C2S WebSocket")

                while not self.stop_event.is_set():
                    try:
                        data = self.data_to_send.get(block=True, timeout=0.005)  # Block for 5ms
                        await websocket.send(data)
                    
                    except queue.Empty:
                        await asyncio.sleep(0.0001)  # Sleep for 100µs to avoid busy-waiting
                    except ConnectionClosed as e:
                        print(f"Connection closed: {e}")
                        break
                    except Exception as e:
                        print(f"Error sending WebSocket message: {e}")

        except Exception as e:
            print(f"Exception in C2S WebSocket client: {e}")


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
            while not self.stop_event.is_set():
                await asyncio.to_thread(target_fn)
                await asyncio.sleep(1 / frequency)
            print("Stopped callback_fn")

        self._timer_threads_callbacks.append(callback_fn())

    def reset_simulation(self):
        message = ResetMsg()
        message.timestamp = self.now()

        wrapped_msg = Any()
        wrapped_msg.type_url = "reset"
        wrapped_msg.value = message.SerializeToString()

        self.data_to_send.put(wrapped_msg.SerializeToString())

    def publish_velocity(self, msg: Twist):
        message = TwistMsg()
        message.timestamp = msg.timestamp
        message.robot_id = msg.robot_id
        message.linear.x = msg.linear[0]
        message.linear.y = msg.linear[1]
        message.angular = msg.angular

        wrapped_msg = Any()
        wrapped_msg.type_url = "vel"
        wrapped_msg.value = message.SerializeToString()

        self.data_to_send.put(wrapped_msg.SerializeToString())

    def now(self):
        return time()
