Download the data
The data can be downloaded here. Next, duplicate or rename data_location_example.txt as data_location.txt and set in it the location of the downloaded gromdata folder.

Note: .vtp files can be inspected with Paraview.

The gromdata contains all the data necessary to train the GNN. However, it is possible to regenerate the data by launching python graph1d/generate_graphs.py from the root of the project.

Train a GNN
From root, type

python network1d/training.py
The parameters of the trained model and hyperparameters will be saved in models, in a folder named as the date and time when the training was launched.

Test a GNN
Within the directory graphs, type

python network1d/tester.py $NETWORKPATH
For example,

python network1d/tester.py models/01.01.1990_00.00.00
This compute errors for all train and test geometries. In the example, models/01.01.1990_00.00.00 is a model generated after training (see Train a GNN).

Some already-trained models are included in gromdata
