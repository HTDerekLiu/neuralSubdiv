## Neural Subdivison
 
The code base only requires standard python dependencies (e.g., numpy). Our code was implemented on python 3.7 with PyTorch 1.3.1. Although not tested on the newer version of PyTorch, welcome to give it a try and let us know if any issues.

For a quick demo, you could use the pre-trained model and test on new shapes. To test the pre-tranied model please run `python test.py /path/to/model/folder/ /path/to/test.obj`. For instance, you can run
```
python test.py ./jobs/net_cartoon_elephant/ ./data_meshes/objs/bunny.obj
```

If you would like to re-train a model, please first generate a dataset in the form of, for instance, `./data_meshes/cartoon_elephant_200/`. 

Once you have the dataset, please run `python gendataPKL.py` to preprocess the meshes into a `.pkl` file, where you need to specify the folder that contains the mesh (please refer to `gendataPKL.py` for more detail).

The nest step is to use `python writeHyperparam.py` to create a folder that contains the parameters of the model. In our example code, runing `python writeHyperparam.py` will create a folder named `./jobs/net_cartoon_elephant/` which contains the model parameters.

Then you can run `python train.py /path/to/model/folder/` to train the model. After training the model, you can use the quick demo code above to test the model.
