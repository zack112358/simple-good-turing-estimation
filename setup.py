import setuptools

with open("README.md", "r") as readme:
    long_description = readme.read()

setuptools.setup(
    name="sgt-zjmccord",
    version="0.0.1",
    author="Zachary McCord",
    author_email="zjmccord@gmail.com",
    description="Simple Good-Turing smoothing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zack112358/simple-good-turing-estimation",
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: BSD License",
                 "Operating System :: OS Independent"],)
