# LSE-FS-UVigo Key-Point Extraction 

LSE-FS-UVigo code to extract, transform and save body-keypoints from
sign-language video datasets. 

## Install

```bash
# clone & enter
git clone https://github.com/lruanova/LSE-FS-UVigo-extraction.git
cd LSE-FS-UVigo-extraction

# create env (Python ≥ 3.12)
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -e .
```

## Usage

The extraction runs in three modes:
- **extract**: extract raw keypoints with mediapipe
- **transform** : apply defined transformations to extracted keypoints
- **all** : extract, then transform
- **single** run on a single video

Example:

```bash
python main.py \
  mode=extract \
  dataset=esaude \
  dataset.dataset_path=/data/ \
```

## Visualizer

Launch with `streamlit run visualizer/app.py`
