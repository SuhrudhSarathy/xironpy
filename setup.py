from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="xiron_py",
    version="0.0.2",
    author="Suhrudh Sarathy",
    author_email="suhrudhsarathy@gmail.com",
    description="Python interface to the Xiron simulator",
    long_description=long_description,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    zip_safe=False,
    include_package_data=True,
    packages=find_packages(where="src"),  # Specify the 'src' directory
    package_dir={"": "src"},  # Specify the root directory for packages
    python_requires=">=3.9",
    install_requires=["websockets", "shapely", "ruff"],
    extras_require={"exp": ["torch", "casadi"]},
)
