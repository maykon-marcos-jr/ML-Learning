read -p "First, ensure you have all packages on deps.list installed. Press Enter to continue..."
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
pyenv install 3.12.13
pyenv virtualenv 3.12.13 .ml_colab_env
pyenv activate .ml_colab_env
pip install --upgrade pip setuptools wheel
TMPDIR=~/.cache pip install -r requirements.txt
python -m ipykernel install --user --name=ml_colab_env --display-name "ML Colab 3.12.13 (venv)"