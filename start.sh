export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
pyenv activate .colab_env
jupyter notebook --no-browser --port=8888 --NotebookApp.allow_origin='*'