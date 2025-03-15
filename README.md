# XironPy
This is a simple Python Interface to use the [Xiron Simulator](https://github.com/SuhrudhSarathy/xiron/tree/main)

# Dependencies
1. Make sure you have python installed on your system. Python can be installed from [here](https://www.python.org/downloads/)
2. Install pip to manage dependencies. Instructions are available [here](https://pip.pypa.io/en/stable/installation/)

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
from time import sleep

from xiron_py.comms import XironContext
from xiron_py.data import Twist


def scan_callback(msg):
    print(f"Recieved Scan message: {msg}")


def pose_callback(msg):
    print(f"Recieved Pose message: {msg}")


def vel_cb():
    msg = Twist(ctx.now(), "robot0", [0.5, 0.0], 0.1)
    ctx.publish_velocity(msg)


if __name__ == "__main__":
    # Create a context object
    ctx = XironContext()

    # Create the Scan Subscriber and add callback function
    ctx.create_scan_subscriber("robot0", scan_callback)

    # Create the Pose Subscriber and add callback function
    ctx.create_pose_subscriber("robot0", pose_callback)

    # Create a timer to publish velocity.
    ctx.create_timer(10, vel_cb)

    # Keep the context alive
    ctx.run()

```

# Documentation
Documentation can be found [here](https://suhrudhsarathy.github.io/xiron/user_guide/python_interface/)