#include <vector>
#include <iostream>
#include <igl>
#include <"dataloader.h">
#include <openmp>
// #include <>


#pragma parallel for
{
    for(i=0; i<folders.size(); i++){
        classname = folders[i];
        train_files = std::dir(classname);
        for(i=0; i<train_files.size(); i++){
            std::cout<<'Reading training '<< classname <<', on '<< i <<' of '<< train_files.size() << '\n';
            std::string file_name = classname + '\train\' + train_files[i];
            igl::load_mesh(file_name, V, F) // trainfile name
            vertices = vertices - repmat(mean(vertices,1),size(vertices,1),1);
            FV.faces = faces;
            #pragma parallel for numofthreads(6){
                for(i=0; i<rotations; i++){
                    th = 2*pi*(j-1)/num_rotations;
                    FV.vertices = [vertices(:,1)*cos(th) - vertices(:,2)*sin(th),vertices(:,1)*sin(th)+vertices(:,2)*cos(th),vertices(:,3)];
                }
            }
            
                
        }
    }
}

# pragma parallel shared
{
    for(i=0; i<folders.size(); i++){
        classname = folders[i];
        train_files = std::dir(classname);
        for(i=0; i<train_files.size(); i++){
            std::cout<<'Reading training '<< classname <<', on '<< i <<' of '<< train_files.size() << '\n';
            std::string file_name = classname + '\train\' + train_files[i];
            igl::load_mesh(file_name, V, F) // trainfile name
            vertices = vertices - repmat(mean(vertices,1),size(vertices,1),1);
            FV.faces = faces;
            for(i=0; i<rotations; i++){
                th = 2*pi*(j-1)/num_rotations;
                FV.vertices = [vertices(:,1)*cos(th) - vertices(:,2)*sin(th),vertices(:,1)*sin(th)+vertices(:,2)*cos(th),vertices(:,3)];
            }
                
        }
    }
}