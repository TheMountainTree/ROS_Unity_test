from setuptools import find_packages, setup

package_name = "eeg_processing"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "scipy",
        "brainda",
    ],
    zip_safe=True,
    maintainer="themountaintree",
    maintainer_email="wang2519480440@outlook.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "central_controller_node = eeg_processing.CentralControllerNode:main",
            "central_controller_ssvep_node = eeg_processing.CentralControllerSSVEPNode:main",
            "central_controller_ssvep_train_node = eeg_processing.CentralControllerSSVEPTrainNode:main",
            "central_controller_ssvep_node2 = eeg_processing.CentralControllerSSVEPNode2:main",
            "central_controller_ssvep_node3 = eeg_processing.CentralControllerSSVEPNode3:main",
            "central_controller_ssvep_node4 = eeg_processing.CentralControllerSSVEPNode4:main",
            "history_sender_node = eeg_processing.history_sender:main",
            "ssvep_communication_node = eeg_processing.SSVEP_Communication_Node:main",
            "ssvep_communication_node2 = eeg_processing.SSVEP_Communication_Node2:main",
            "ssvep_communication_node3 = eeg_processing.SSVEP_Communication_Node3:main",
            "ssvep_communication_node3_1 = eeg_processing.SSVEP_Communication_Node3_1:main",
        ],
    },
)
