from setuptools import setup, find_packages

setup(
    name='noise_dr',
    version='1.0',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    license='Apache 2.0',
    python_requires='>=3.7',
    install_requires=[
        "torch",
        "transformers",
        "textattack",
        "datasets",
        "faiss-cpu",
        "numpy"
    ]
)
