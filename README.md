# XironPy
This is a simple Python Interface to use the [Xiron Simulator](https://github.com/SuhrudhSarathy/xiron/tree/main)

# Installation
1. Use pip to install the module. To install it directly from github:
```
pip install git+https://github.com/SuhrudhSarathy/xironpy.git
```
2. To install from source, clone the repository and use pip to install
```
git clone https://github.com/SuhrudhSarathy/xironpy.git
```
```
cd xironpy
pip install -e .
```

# Usage
A simple script can be found in [examples](./src/xiron_py/examples/connection_test.py)

```python
from xiron_py.comms import XironContext
from xiron_py.data import Twist
from time import sleep


def scan_callback(msg):
    print(f"Recieved Scan message: {msg}")


def pose_callback(msg):
    print(f"Recieved Pose message: {msg}")


if __name__ == "__main__":
    # Create a context object
    ctx = XironContext()

    # Create the Velocity publisher for robot0
    vel_pub = ctx.create_vel_publisher("robot0")

    # Create the Scan Subscriber and add callback function
    ctx.create_scan_subscriber("robot0", scan_callback)

    # Create the Pose Subscriber and add callback function
    ctx.create_pose_subscriber("robot0", pose_callback)

    twist_message = Twist("robot0", [0.1, 0.0], 0.1)
    for i in range(100):
        vel_pub.publish(twist_message)
        print("Publihsed vel: ", i)
        sleep(0.1)

    twist_message = Twist("robot0", [0.0, 0.0], 0.0)
    vel_pub.publish(twist_message)

    print("Done!")

```

# Documentation
Documentation can be found [here](https://suhrudhsarathy.github.io/xiron/user_guide/python_interface/)