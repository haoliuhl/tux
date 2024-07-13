from setuptools import find_packages, setup

setup(
    name='tux',
    version='0.0.3',
    license='MIT',
    description='Tools and Utils for JAX/Flax',
    url='https://github.com/forhaoliu/tux',
    packages=find_packages(include=['tux']),
    python_requires=">=3.7",
    install_requires=[
        'absl-py',
        'ml-collections',
        'wandb',
        'gcsfs',
        'cloudpickle',
        'numpy',
        'transformers',
        'jax',
        'flax',
        'optax',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
)
