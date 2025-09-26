# PYCFD

[![CI - Tests](https://github.com/lukbrb/pycfd/actions/workflows/ci.yml/badge.svg)](https://github.com/lukbrb/pycfd/actions/workflows/ci.yml)
[![Lint & Typecheck](https://github.com/lukbrb/pycfd/actions/workflows/lint.yml/badge.svg)](https://github.com/lukbrb/pycfd/actions/workflows/lint.yml)
[![Pre-commit](https://github.com/lukbrb/pycfd/actions/workflows/lint.yml/badge.svg)](https://github.com/lukbrb/pycfd/actions/workflows/lint.yml)

Bac à sable en `Python` pour expérimenter des schémas numériques pour la CFD.
Ce code n'a pas pour but d'être utilisé pour de vraies simulations.

## Installation

Commencez par cloner le répertoire 
```console
git clone https://github.com/lukbrb/pycfd.git && cd pycfd
````

## Dépendences

Le plus simple, et l'option recommandée, est d'utiliser le gestionnaire de paquet et projet [uv](https://docs.astral.sh/uv/guides/install-python/). Une fois installé, il suffira de faire

```console
uv run main.py
```

afin de lancer une simulation.

> [!IMPORTANT]  
> Avant de lancer une simulation, assurez vous de choisir les paramètres de votre simulation.  
> Pour l'instant, les paramètres sont modifiables dans le fichier `src/params`.  