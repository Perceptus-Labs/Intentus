from setuptools import setup, find_packages


def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="intentus",
    version="0.1.0",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
    author="Haohan Wang",
    author_email="your.email@example.com",
    description="An SDK for robotics interaction with audio and video processing capabilities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/intentus",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "intentus": ["py.typed"],
    },
)
