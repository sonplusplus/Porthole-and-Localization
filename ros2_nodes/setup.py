from setuptools import setup


setup(
    name="ev_pothole_localization_nodes",
    version="0.1.0",
    py_modules=["pothole_node", "localization_node"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/ev_pothole_localization_nodes"]),
        ("share/ev_pothole_localization_nodes", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="EV Pothole Team",
    maintainer_email="maintainer@example.com",
    description="ROS 2 interface stubs for pothole detection and localization outputs.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "pothole_node = pothole_node:main",
            "localization_node = localization_node:main",
        ],
    },
)
