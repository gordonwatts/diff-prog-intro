These jupyter notebooks are published on github pages [here](https://gordonwatts.github.io/diff-prog-intro).

# Development

Because this uses JAX, it can't run on windows directly (you have to get XLA running there). Use WSL instead.

To get the GPU version installed on windows:

```
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Do that after doing the `pip install -r book/requirements.txt`.

## Publishing:

```bash
jupyter-book build book/
cd book
ghp-import -n -p -f _build/html
```

Make sure to check for errors after the build - if your terminal window is small they often scroll up out of sight!