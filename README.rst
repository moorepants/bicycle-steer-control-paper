This is a working paper that explores the theoretical possibilities and
limitations of adding a motor between the rear and front frames of a bicycle.
This motor applies a steering torque that is synonymous to the effective torque
applied by the rider when controlling a bicycle.

An HTML version can be viewed at:

https://moorepants.github.io/bicycle-steer-control-paper

Contents of this repository are licensed as CC-BY 4.0, see the ``LICENSE``
file.

Install dependencies::

   conda env create -f bicycle-steer-control-paper.yml
   conda activate bicycle-steer-control-paper
   python -m pip install git+https://github.com/moorepants/BicycleParameters

Run the scripts, e.g.::

  python src/pd_control.py

Build the figures and paper::

  make pdf

Run the notebook::

   jupyter notebook
