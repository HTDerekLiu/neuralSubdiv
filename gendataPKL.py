from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from include import *

def main():
    mesh_folders = ['./data_meshes/cartoon_elephant_200/']
    # mesh_folders = ['./data_meshes/bunny/', './data_meshes/rockerArm/', './data_meshes/fertility/']
    S = Categories(mesh_folders)

    pickle.dump(S, file = open("./data_PKL/cartoon_elephant_train.pkl", "wb"))
if __name__ == '__main__':
    main()