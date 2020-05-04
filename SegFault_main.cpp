#include <iostream>
#include <Python.h>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

int main()
{
	char filename[] = "SegFault_train_VAE.py";
	File* pyfile;

	//Open a python environment
	Py_Initialize();

	pyfile = _Py_fopen(filename, "r");
	//Run the python file, generates a .txt file in the directory
	PyRun_SimpleFile(pyfile, filename);

	

	//std::vector<std::vector<std::vector<std::vector<int>>>> voxelData; //4d vector of dimensions (voxel index, x, y, z)

	vector<vector<vector<vector<int> > > > voxelData(
		3991,
		vector < vector < vector<int>(
			30,
			vector < vector<int>(
				30,
				vector<int>(30)
				)
			)
	);

	ifstream fin;
	fin.open("Generated_Voxels.txt")

	for (int v = 2; v++; v < 3991) //Data starts on third line. Iterates through lines.
	{
		for (int i = 0; i++; i < 30)
		{
			for (int j = 0; j++; j < 30)
			{
				for (int k = 0; k++; k < 30)
				{
					fin >> voxelData[v][i][j][k];
				}
			}
		}
	}
	
	// int index = random_select() 
	// putIntoGUI( &voxelData[index] ) ;
	//Run voxeldata into GUI file as parameter
	Py.Finalize();
	return 0;
}


