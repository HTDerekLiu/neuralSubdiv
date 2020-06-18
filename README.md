## Neural Subdivison
 
[Neural subdivision](https://www.dgp.toronto.edu/projects/neural-subdivision/) subdivides a triangle mesh using neural networks. This is a prototype implementation in Python 3.7  with PyTorch 1.3.1 and MATLAB. The Python code only requires standard dependencies (e.g., numpy), and the MATLAB code depends on [gptoolbox](https://github.com/alecjacobson/gptoolbox).

For a quick demo, please use the pre-trained model and test on new shapes. To test the pre-tranied model please run `python test.py /path/to/model/folder/ /path/to/test.obj`. For instance, you can run
```
python test.py ./jobs/net_cartoon_elephant/ ./data_meshes/objs/bunny.obj
```

If you would like to re-train a model, please first generate a dataset in the form of, for instance, `./data_meshes/cartoon_elephant_200/`. This could be done by running the MATLAB script `genTrainData_slow.m`.

Once you have the dataset, please run `python gendataPKL.py` to preprocess the meshes into a `.pkl` file, where you need to specify the folder that contains the mesh (please refer to `gendataPKL.py` for more detail).

The next step is to use `python writeHyperparam.py` to create a folder that contains the parameters of the model (see `writeHyperparam.py` for more detail). In our example code, running `python writeHyperparam.py` will create a folder named `./jobs/net_cartoon_elephant/` which contains the model parameters.

Then you can run `python train.py /path/to/model/folder/` to train the model. For instance, with the default folder generated with the above script, you can simply run `python train.py ./jobs/net_cartoon_elephant/` to train the model. After training, you can use the quick demo code `test.py` to test the model by running `python test.py /path/to/model/folder/ /path/to/testMesh.obj`.

If any questions, please contact Hsueh-Ti Derek Liu (hsuehtil@cs.toronto.edu). 
