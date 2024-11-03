import json
import asyncio
import websockets
from dataclasses import asdict
from threading import Thread, Lock
from xiron_py.data import LaserScan, Pose, Twist

_mutex = Lock()


class XironContext:
    def __init__(self):
        self.robot_name_scan_subscriber_map = {}
        self.robot_name_pose_subscriber_map = {}
        self.websocket = None
        self.loop = asyncio.new_event_loop()

        # Start the WebSocket connection
        self.loop.run_until_complete(self._connect())

        # Start the message handling thread
        self._message_thread = Thread(target=self._run_event_loop, daemon=True)
        self._message_thread.start()

    async def _connect(self):
        self.send_websocket = await websockets.connect("ws://localhost:8765")
        self.recv_websocket = await websockets.connect("ws://localhost:8766")

        print("Connected to websockets")

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._handle_messages())

    async def _handle_messages(self):
        print("Listeneing for messages")
        while True:
            try:
                async for message in self.recv_websocket:
                    print(message)
                    data = json.loads(message)
                    if "type" in data:
                        if data["type"] == "scan":
                            self._handle_scan(data)
                        elif data["type"] == "pose":
                            self._handle_pose(data)
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed. Attempting to reconnect...")
                await self._connect()

    def _handle_scan(self, data):
        scan = LaserScan(
            data["robot_id"],
            data["angle_min"],
            data["angle_max"],
            data["num_readings"],
            data["values"],
        )
        callback_function = self.robot_name_scan_subscriber_map.get(scan.robot_id)
        if callback_function:
            callback_function(scan)

    def _handle_pose(self, data):
        pose = Pose(data["robot_id"], data["position"], data["orientation"])
        callback_function = self.robot_name_pose_subscriber_map.get(pose.robot_id)
        if callback_function:
            callback_function(pose)

    async def _send_velocity(self, data: Twist):
        with _mutex:
            message = json.dumps({"type": "velocity", "data": asdict(data)})
            await self.send_websocket.send(message)

    async def _send_reset(self):
        with _mutex:
            message = json.dumps({"type": "reset"})
            await self.websocket.send(message)

    def create_vel_publisher(self, robot_id: str):
        return VelPublisher(robot_id=robot_id, ctx=self)

    def create_pose_subscriber(self, robot_id: str, callback_fn):
        self.robot_name_pose_subscriber_map[robot_id] = callback_fn

    def create_scan_subscriber(self, robot_id: str, callback_fn):
        self.robot_name_scan_subscriber_map[robot_id] = callback_fn

    def reset_simulation(self):
        print("Resetting Simulation")
        asyncio.run_coroutine_threadsafe(self._send_reset(), self.loop)


class VelPublisher:
    def __init__(self, robot_id: str, ctx: XironContext):
        self._ctx = ctx
        self.robot_id = robot_id

    def publish(self, message: Twist):
        asyncio.run_coroutine_threadsafe(
            self._ctx._send_velocity(message), self._ctx.loop
        )
