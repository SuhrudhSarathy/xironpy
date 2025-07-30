from xiron_py.controller.pure_pursuit import PurePursuitController
import numpy as np

pp = PurePursuitController()


plan = np.random.randn(3, 100)
pp.set_target_plan(plan)

current_pose = np.zeros((3, 1))

output_path = pp.transform_path(current_pose)
print(output_path)