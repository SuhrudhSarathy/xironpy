from xiron_py.env import EnvironmentManager

if __name__ == "__main__":
    env = EnvironmentManager("src/xiron_py/examples/basics/config.yaml")
    path = [[-4.0, 4.0], [-1.0, 6.0]]
    point = [-2.0, 3.1]
    point2 = [1.0, 2.93]

    print(env.isPathValid(path))
    print(env.isPoseWithin(point))
    print(env.distance(point2))
