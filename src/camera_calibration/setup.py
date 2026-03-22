from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'camera_calibration'

# 自动收集 nodes/ 目录下所有 .py 脚本（排除 __init__.py）
nodes = []
nodes_dir = package_name + '/nodes'
if os.path.isdir(nodes_dir):
    for file in os.listdir(nodes_dir):
        if file.endswith('.py') and file != '__init__.py':
            script_name = file[:-3]  # 移除 .py 后缀
            # 格式: '脚本名 = 包路径.模块:main函数'
            nodes.append(f'{script_name} = {package_name}.nodes.{script_name}:main')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wj',
    maintainer_email='3199320251@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': nodes,
    },
)
