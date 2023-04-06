import setuptools

setuptools.setup(
    name='pcsr',
    version='0.1',
    author='Nithin and Joel',
    description='PCSR data structure to represent Dynamic Graph Representation',
    url='https://github.com/GNN-Compiler/Seastar',
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.5',
    zip_safe=False,
    packages=['pcsr'],
    package_dir={'pcsr': 'pcsr'},
    package_data={'pcsr': ['pcsr.so']}
)