#include <iostream>
#include <Python.h>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

std::vector<std::vector<std::vector<int>>> getInference(){
	// std::string model_file = "SegFault_train_VAE.py";

	//Open a python environment
	Py_Initialize();
	std::cout<<"[deg] Strating\n";
	PyObject *pFunc, *pArgs, *pModule, *input, *pDict, *pResult;
	
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append(\".\")");

	pModule = PyImport_Import(PyString_FromString("trial"));
	std::cout<<"[deg] find m\n";
	if (!pModule) {
        PyErr_Print();
        exit(1);
    }
	pDict = PyModule_GetDict(pModule);
	std::cout<<"[deg] get dict\n";
    
	std::cout<<"[deg] func\n";
	pFunc = PyDict_GetItemString(pDict, "main");
	//Run the python file, generates a .txt file in the directory
	pArgs = PyTuple_New(1);
	// PyObject* pointsLeft = vectorToList_Float(v_left);
	std::cout<<"[deg] build\n";
	input = Py_BuildValue("i", 42);
	std::cout<<"[deg] build bnew\n";
	PyTuple_SetItem(pArgs, 0, input);
	pResult = PyObject_CallObject(pFunc, pArgs);

	std::vector<std::vector<std::vector<std::vector<int>>>> voxelData; //4d vector of dimensions (voxel index, x, y, z)
	// voxelData.resize(3991);
	// std::vector<std::vector<std::vector<std::vector<int> > > > voxelData(
	// 	3991,
	// 	std::vector < std::vector < std::vector<int>(
	// 		30,
	// 		std::vector < std::vector<int>(
	// 			30,
	// 			std::vector<int>(30)
	// 			)
	// 		)
	// );

	std::string filename = "Generated_Voxels.txt";
	std::ifstream filestream(filename.c_str());

	voxelData.resize(3991);
	for (int v = 2; v++; v < 3991) //Data starts on third line. Iterates through lines.
	{
		voxelData[v].resize(30);
		for (int i = 0; i++; i < 30)
		{
			voxelData[i].resize(30);
			for (int j = 0; j++; j < 30)
			{
				voxelData[j].resize(30);
				for (int k = 0; k++; k < 30)
				{
					voxelData[k].resize(30);
					filestream >> voxelData[v][i][j][k];
				}
			}
		}
	}
	
	// int index = random_select() 
	// putIntoGUI( &voxelData[index] ) ;
	//Run voxeldata into GUI file as parameter
	Py_Finalize();
	return voxelData[0];
}
int main(){
	getInference();
}




