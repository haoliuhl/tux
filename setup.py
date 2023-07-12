from setuptools import find_packages, setup

setup(
    name='tux',
    version='0.0.1',
    license='MIT',
    description='Tools and Utils. Some tools and utils modified from many other code to fit my needs.',
    url='https://github.com/lhao499/tux',
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
    ],
    extras_require={
        'jax': [
            'jax',
            'flax',
            'optax',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
)
