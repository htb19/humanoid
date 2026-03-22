from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'robot_simulation'

def generate_data_files(package_dir, install_dir):
    """
    递归生成 data_files 条目，保留目录结构。
    :param package_dir: 源目录（如 'models'）
    :param install_dir: 安装根目录（如 'share/robot/models'）
    :return: list of (destination, [source_files])
    """
    data_files = []
    for root, dirs, files in os.walk(package_dir):
        # 计算当前 root 相对于 package_dir 的子路径
        rel_path = os.path.relpath(root, package_dir)
        if rel_path == '.':
            # 根目录
            dest = install_dir
        else:
            # 子目录，如 'urdf' -> install_dir/urdf
            dest = os.path.join(install_dir, rel_path)
        
        # 获取当前目录下所有文件的完整路径
        source_files = [os.path.join(root, f) for f in files]
        if source_files:  # 只有非空才添加
            data_files.append((dest, source_files))
    return data_files
    
# 生成 models/ 的安装条目
models_files = generate_data_files(
    package_dir='models',
    install_dir=os.path.join('share', package_name, 'models')
)

# 自动收集 nodes/ 目录下所有 .py 脚本（排除 __init__.py）
nodes = []
nodes_dir = 'robot_simulation/nodes'
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
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        *models_files,
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
