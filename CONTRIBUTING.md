# Contributing to `ipyml`

## Pre-requisites

> ### Windows Users
>
> Please try to put your base `conda` and your git checkout in the shortest possible
> paths, and on the same, local drive, e.g. `c:\mc3` and `c:\git\ipyml`. Try to
> avoid paths managed by DropBox, OneDrive, etc. and consider turning off search
> indexing and Windows Defender for these paths.
>
> Also, you may wish to ensure you have no existing Jupyter kernels in your user paths:
> basically anything in the output of `jupyter --paths` that looks like
> HOME/AppData/something can be safely deleted. They will be recreated
> as needed with the proper permissions.

- install [Mambaforge](https://github.com/conda-forge/miniforge/releases)
- install `anaconda-project` and `doit` into the `base` env

```bat
mamba install -c conda-forge anaconda-project doit
```

or, use the same base environment as CI:

```bat
:: windows
mamba env update --file .ci\environment.yml
c:\mc3\envs\base-ipyml\Scripts\activate
```

```bash
# unix
mamba env update --file .ci/environment.yml
source ~/mc3/envs/base-ipyml/bin/activate
```

## Installing the development environment with `anaconda-project`

```bat
(base-ipyml) anaconda-project run
```

This will create the necessary environments to run commands.

## Running lab

```bat
(base-ipyml) jupyter lab
```

This will open lab, where you can run code/examples.