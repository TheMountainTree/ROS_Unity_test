import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/themountaintree/workspace/ROS_Unity_test/install/eeg_processing'
