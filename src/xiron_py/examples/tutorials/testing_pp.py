from xiron_py.comms import XironContext
from xiron_py.data import Pose, Twist
from xiron_py.controller.pure_pursuit import PurePursuitController

ctx = XironContext()
ctx.run_in_separate_thread()