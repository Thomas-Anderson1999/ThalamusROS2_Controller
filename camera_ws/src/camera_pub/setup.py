from setuptools import find_packages, setup

package_name = 'camera_pub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kunbum-park',
    maintainer_email='kunbum.park@groove-x.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'record_node = camera_pub.record_node:main',
            'camera_node_d455i = camera_pub.camera_node_d455i:main',
            'image_viewer = camera_pub.image_viewer:main',
        ],
    },
)
