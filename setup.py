import setuptools

setuptools.setup(
    name='seastar',
    version='1.0.0',
    author='Joel Mathew Cherian, Nithin Puthalath Manoj',
    author_email='joelmathewcherian@gmail.com, nithinp.manoj@gmail.com',
    description='Vertex-Centric Approach to building Graph Neural Networks',
    url='https://github.com/bfGraph/Seastar',
    packages=setuptools.find_packages(),
    package_data={
        "": ["*.so","*.jinja"]
    },
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
