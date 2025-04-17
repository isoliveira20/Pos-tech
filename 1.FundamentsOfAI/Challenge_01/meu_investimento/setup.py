from setuptools import setup

setup(
    name="meu_investimento",
    version="0.1.0",
    description="Pacote para cálculos de investimentos.",
    author="Izabela Oliveira",
    author_email="izabela.oliveira@example.com",
    packages=["investimentos"],
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0"
    ],
)
