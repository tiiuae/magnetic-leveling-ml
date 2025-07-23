This repository contains the code for the paper "Deep learning-based approach for Leveling Airborne Magnetic Data".


##Setup Instructions

1. Clone the repository:
-------------------------
git clone https://github.com/tiiuae/magnetic-leveling-ml.git
cd magnetic-leveling-ml

2. Create and activate the Conda environment:
---------------------------------------------
conda create -n maglev python=3.10 -y
conda activate maglev

3. Install dependencies:
------------------------
    pip install -r requirements.txt


##Running the Scripts

1. To train the model:
-----------------------
python train.py

2. To test the model:
----------------------
python test.py


========================================
Notes
========================================

- Make sure any required datasets or configuration files are placed in the correct directories.
- If the scripts accept command-line arguments, modify the command accordingly.
- Activate the conda environment (`conda activate maglev`) before running the scripts.
