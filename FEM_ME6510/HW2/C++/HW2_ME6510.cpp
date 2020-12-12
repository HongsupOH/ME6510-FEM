// HW2_ME6510.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <Eigen>
#include "Truss.h"

using namespace Eigen;
using namespace std;

int main()
{
    //Define f, s, force, deform vectors
    VectorXd f(2);
    f(0) = 0;
    f(1) = 1;
    VectorXd s(6);
    s(0) = 2;
    s(1) = 3;
    s(2) = 4;
    s(3) = 5;
    s(4) = 6;
    s(5) = 7;
    VectorXd force(2);
    force(0) = 1000;
    force(1) = -1000;
    VectorXd deform(6);
    deform(0) = 0.0;
    deform(1) = 0.0;
    deform(2) = 0.0;
    deform(3) = 0.0;
    deform(4) = 0.0;
    deform(5) = 0.0;
    //Define problem
    int num_node = 4;
    int num_ele = 3;
    double E = 10e6;
    double A = 1;
    //Create problem
    Truss<2> Tr(num_node, num_ele, E, A);
    //Add node coordinate
    Tr.add_node(0, 0, 0);
    Tr.add_node(1, 100 * sqrt(3), 100);
    Tr.add_node(2, 0, 100);
    Tr.add_node(3, 100 / sqrt(3), 100);
    cout << "Node Coordinate: " << endl;
    cout<< Tr.node_coor << endl;
    //Add elem connection
    Tr.add_elem(0, 0, 1);
    Tr.add_elem(1, 0, 2);
    Tr.add_elem(2, 0, 3);
    cout << "Element information: " << endl;
    cout << Tr.elem_conn << endl;
    //Add dof information
    Tr.add_dofs();
    cout << "Degrees of freedom: " << endl;
    cout << Tr.node_dofs << endl;
    //Add theta
    Tr.add_theta(60);
    Tr.add_theta(0);
    Tr.add_theta(-30);
    //GlobalK
    Tr.globK();
    cout << "Globla K: " << endl;
    cout << Tr.K << endl;
    //Solve
    int f_ind[] = {0,1};
    int d_ind[] = {2,3,4,5,6,7};
    Tr.solve(force, deform,f_ind,d_ind);
    cout << "Answer: " << endl;
    cout << "D_{f}: " << endl;
    cout << Tr.D_f << endl; 
    cout << "F_{s}: " << endl;
    cout << Tr.F_s << endl;
}
