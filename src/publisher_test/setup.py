from setuptools import find_packages, setup
import os
from glob import glob

package_name = "publisher_test"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="themountaintree",
    maintainer_email="user@example.com",
    description="Test publisher for Unity ROS2 TCP connection",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "image_publisher = publisher_test.image_publisher:main",
            "seg_image_publisher = publisher_test.seg_image_publisher:main",
            "udp_sender_node = publisher_test.udp_sender_node:main",
            "eeg_tcp_listener_node = publisher_test.eeg_tcp_listener_node:main",
            "reasoner_publish_test = publisher_test.reasoner_publish_test:main",
            "reasoner_publish_test_1 = publisher_test.reasoner_publish_test_1:main",
            "reasoner_publish_test_2 = publisher_test.reasoner_publish_test_2:main",
            "reasoner_publish_test_2_local_llm = publisher_test.reasoner_publish_test_2_local_llm:main",
        ],
    },
)
