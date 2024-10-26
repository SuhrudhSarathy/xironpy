import json
from dataclasses import asdict
from threading import Thread

import zmq

from xiron_py.data import LaserScan, Pose, Twist


class XironContext:
    def __init__(self):
        self._ctx = zmq.Context()

        self.robot_name_scan_subscriber_map = {}
        self.robot_name_pose_subsriber_map = {}

        # Add the main Scan subscriber, Pose Subscriber and Vel publisher
        self.scan_sub = self._ctx.socket(zmq.SUB)
        self.scan_sub.connect("tcp://127.0.0.1:5858")
        self.scan_sub.subscribe(b"scan")

        # Pose Sub
        self.pose_sub = self._ctx.socket(zmq.SUB)
        self.pose_sub.connect("tcp://127.0.0.1:5555")
        self.pose_sub.subscribe(b"pose")

        # Vel pub
        self.vel_pub = self._ctx.socket(zmq.PUB)
        self.vel_pub.bind("tcp://127.0.0.1:5556")

        self.vel_topic = (
            b"vel"  # The topic for the message (can be any bytes-like object)
        )

        self.reset_pub = self._ctx.socket(zmq.PUB)
        self.reset_pub.bind("tcp://127.0.0.1:5956")

        self.reset_topic = b"reset"

        # Start the threads
        self._pose_callback_thread = Thread(
            target=self._main_pose_data_sub, daemon=True
        )
        self._pose_callback_thread.start()

        self._scan_callback_thread = Thread(
            target=self._main_scan_data_sub, daemon=True
        )
        self._scan_callback_thread.start()

    def _main_scan_data_sub(self):
        while True:
            output = self.scan_sub.recv_string(0)
            if output != "scan":
                json_obj = json.loads(output)
                data = LaserScan(
                    json_obj["robot_id"],
                    json_obj["angle_min"],
                    json_obj["angle_max"],
                    json_obj["num_readings"],
                    json_obj["values"],
                )

                callback_function = self.robot_name_scan_subscriber_map.get(
                    data.robot_id
                )

                if callback_function is not None:
                    callback_function(data)

    def _main_pose_data_sub(self):
        while True:
            output = self.pose_sub.recv_string(0)
            if output != "pose":
                json_obj = json.loads(output)
                data = Pose(
                    json_obj["robot_id"], json_obj["position"], json_obj["orientation"]
                )

                callback_function = self.robot_name_pose_subsriber_map.get(
                    data.robot_id
                )

                if callback_function is not None:
                    callback_function(data)

    def _send_velocity(self, data: Twist):
        vel_string = json.dumps(asdict(data))
        message = vel_string.encode(
            "utf-8"
        )  # The message to send (can be any bytes-like object)

        # Send the message with the specified topic
        self.vel_pub.send_multipart([self.vel_topic, message])

    def _send_reset(self):
        message = "reset".encode("utf-8")
        self.reset_pub.send_multipart([self.reset_topic, message])

    def create_vel_publisher(self, robot_id: str):
        return VelPublisher(robot_id=robot_id, ctx=self)

    def create_pose_subscriber(self, robot_id: str, callback_fn):
        self.robot_name_pose_subsriber_map[robot_id] = callback_fn

    def create_scan_subscriber(self, robot_id: str, callback_fn):
        self.robot_name_scan_subscriber_map[robot_id] = callback_fn

    def reset_simulation(self):
        print("Resetting Simulation")
        self._send_reset()


class VelPublisher:
    def __init__(self, robot_id: str, ctx: XironContext):
        self._ctx = ctx
        self.robot_id = robot_id

    def publish(self, message: Twist):
        self._ctx._send_velocity(message)
