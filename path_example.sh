export PYTHONPATH=$PYTHONPATH:/home/<username>/<projectParentpath>
# Copy this file to path.sh and fill in the placeholders below,
# then `source path.sh` to set up PYTHONPATH and the venv.

source < venv-path >/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True