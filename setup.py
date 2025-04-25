"""
    Setup file for pyReCo.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.6.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""

from setuptools import setup, find_packages

if __name__ == "__main__":
    try:
        setup(
            name="pyreco",
            version="0.1.0",
            packages=find_packages(where="src"),
            package_dir={"": "src"},
            install_requires=[
                "numpy",
                "scikit-learn",
                "scikit-optimize",
                "pyyaml",
            ],
            extras_require={
                "dev": [
                    "pytest",
                    "pytest-cov",
                    "black",
                    "flake8",
                ],
            },
            python_requires=">=3.7",
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
