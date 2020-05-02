#include <vector>
#include <iostream>
#include <igl>

Eigen::MatrixXd V;
Eigen::MatrixXi F;


class Dataloader(){
    public:
        bool checkAvailable(Eigen::Matrix );
        Eigen::Matrix getFaces(Eigen::Matrix);
        Eigen::Matrix getVertices(Eigen::Matrix);
};