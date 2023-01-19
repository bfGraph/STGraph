import setuptools

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
