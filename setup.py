import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "iirsBenchmark",
    version = "1.1.2",
    author = "Guilherme Aldeia",
    packages = setuptools.find_packages(
        include = ['./iirsBenchmark'],
        exclude = ['./experiments'],
    ),
    author_email = "guilherme.aldeia@ufabc.edu.br",
    description = "Interpretability in Symbolic Regression: a Benchmark of methods",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    python_requires = ">=3.7",
    license = "BSD-3-Clause",
    install_requires = [
        # utility modules
        "numpy==1.22.0",
        "filelock==3.0.12",
        "scipy==1.10.0",
        "jax==0.2.13",
        "jaxlib==0.1.67",

        # regressor modules
        "scikit-learn==0.24.2",
        "itea==1.1.0",

        # explainer modules (fixed versions due deprecation warnings of its dependences)
        "shap==0.40.0",
        "sage-importance==0.0.4",
        "interpret==0.2.7",
        "lime==0.2.0.1",
    ],
    setup_requires = ["wheel", "pytest-runner"],
    tests_require = [
        "pytest",
        "pytest-html",
        "pytest-xdist"
    ],
    test_suite = "tests",
)