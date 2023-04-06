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

setuptools.setup(
    name='seastar_graph',
    version='0.1',
    author='Nithin and Joel',
    description='Graph Abstraction for Seastar',
    packages=setuptools.find_packages(include=['seastar_graph', 'seastar_graph.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    zip_safe=False
)

setuptools.setup(
    name='seastar',
    version='0.1',
    author='ydwu',
    author_email='ydwu2014@gmail.com',
    description='Easy Fast Graph learning',
    url='https://github.com/ydwu4/seastar-paper-version',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
        'Jinja2>=2',
        'pynvrtc',
        'pydot',
        'networkx',
        'sympy'
        ],
    zip_safe=False
)