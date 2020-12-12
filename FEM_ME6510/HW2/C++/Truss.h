#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include<Eigen>
# define M_PI           3.14159265358979323846  /* pi */
using namespace Eigen;
using namespace std;

template<int dim>
class Truss {
	//Constructor
public:
	Truss(int Num_node, int Num_ele, double Youngs, double Area);
	//Deconstructor
	~Truss();
	//Declare Functions
	void add_node(int node, double x, double y);
	void add_elem(int ele, int n1, int n2);
	void add_dofs();
	void add_theta(double theta);
	MatrixXd truss_stiffness(double theta, double E, double A, double L);
	void globK();
	void solve(VectorXd F_f, VectorXd D_s,int f[], int d[]);

	//Declare Variables
	int num_node, num_ele;
	int num_dofs;
	double E, A;
	vector <double> thetas;
	MatrixXd K;
	MatrixXd node_coor;
	MatrixXd elem_conn;
	MatrixXd node_dofs;
	MatrixXd D_f;
	MatrixXd F_s;
};

//Class Constructor
template<int dim>
Truss<dim>::Truss(int Num_node, int Num_ele, double Youngs, double Area) {
	num_node = Num_node;
	num_ele = Num_ele;
	num_dofs = num_node * dim;
	E = Youngs;
	A = Area;
	K.resize(num_dofs, num_dofs);
	K=MatrixXd::Zero(num_dofs, num_dofs);
	node_coor.resize(num_node, 2);
	elem_conn.resize(num_ele, 2);
	node_dofs.resize(num_node, 2);
}
//Class destructor
template<int dim>
Truss<dim>::~Truss() {
}
//Add node coordinate
template <int dim>
void Truss<dim>::add_node(int node, double x, double y) {
	node_coor(node, 0) = x;
	node_coor(node, 1) = y;
}
//Add element information
template <int dim>
void Truss<dim>::add_elem(int ele, int n1, int n2) {
	elem_conn(ele, 0) = n1;
	elem_conn(ele, 1) = n2;
}
//Add degree of freedom
template <int dim>
void Truss<dim>::add_dofs() {
	for (unsigned int i = 0; i < num_node; i++) {
		node_dofs(i, 0) = 2 * i;
		node_dofs(i, 1) = 2 * i + 1;
	}
}
//Add theta
template <int dim>
void Truss<dim>::add_theta(double theta) {
	thetas.push_back(theta);
}
//Element K
template <int dim>
MatrixXd Truss<dim>::truss_stiffness(double theta, double E, double A, double L) {
	double theta_r = theta * M_PI / 180.0;
	double k = E * A / L;
	MatrixXd k_loc;
	k_loc.resize(4, 4);
	k_loc<< k, 0, -k, 0,
		0, 0, 0, 0,
		-k, 0, k, 0,
		0, 0, 0, 0;
	double C = cos(theta_r);
	double S = sin(theta_r);
	MatrixXd Tr;
	Tr.resize(4, 4);
	Tr << C, S, 0, 0,
		-S, C, 0, 0,
		0, 0, C, S,
		0, 0, -S, C;
	MatrixXd k_ele = (Tr.transpose()*k_loc)*Tr;
	return k_ele;
}
//Generate global K
template<int dim>
void Truss<dim>::globK() {
	for (int e = 0; e < num_ele; e++) {
		MatrixXd elem_nodes = elem_conn.row(e);
		MatrixXd elem_node_coors(2, node_coor.row(elem_nodes(0)).size());
		elem_node_coors.row(0) = node_coor.row(elem_nodes(0));
		elem_node_coors.row(1) = node_coor.row(elem_nodes(1));
		double dif_0 = elem_node_coors(0, 0) - elem_node_coors(1, 0);
		double dif_1 = elem_node_coors(0, 1) - elem_node_coors(1, 1);
		double L_ele = sqrt(dif_0 * dif_0 + dif_1 * dif_1);
		double theta = thetas.at(e);
		MatrixXd K_ele = truss_stiffness(theta, E, A, L_ele);
		double n1 = elem_nodes(0);
		double n2 = elem_nodes(1);
		MatrixXd dof1 = node_dofs.row(n1);
		MatrixXd dof2 = node_dofs.row(n2);
		MatrixXd cur_dofs(dof1.size(), dof2.size());
		cur_dofs << dof1,dof2;
		MatrixXd cur_dof = cur_dofs.transpose();
		cout << "dofs "<<cur_dofs << endl;
		for (int i = 0; i < cur_dof.size(); i++) {
			for (int j = 0; j < cur_dof.size(); j++) {
				int loc1 = i;
				int loc2 = j;
				int glb1 = cur_dof(i);
				int glb2 = cur_dof(j);
				double cur = K(glb1, glb2);
				double update = K(glb1,glb2)+ K_ele(loc1, loc2);
				K(glb1, glb2) = update;
			}
		}
	}
}
//Solve
template<int dim>
void Truss<dim>::solve(VectorXd F_f, VectorXd D_s,int f[], int d[]) {
	MatrixXd K_ff(F_f.size(), F_f.size());
	MatrixXd K_fs(F_f.size(), D_s.size());
	MatrixXd K_sf(D_s.size(), F_f.size());
	MatrixXd K_ss(D_s.size(), D_s.size());
	//K_ff
	for (unsigned int i = 0; i < F_f.size(); i++) {
		for (unsigned int j = 0; j < F_f.size(); j++) {
			int gl1 = f[i];
			int gl2 = f[j];
			K_ff(i, j) = K(gl1, gl2);
		}
	}
	//K_fs and K_sf
	for (unsigned int i = 0; i < F_f.size(); i++) {
		for (unsigned int j = 0; j < D_s.size(); j++) {
			int gl1 = f[i];
			int gl2 = d[j];
			K_fs(i, j) = K(gl1, gl2);
			K_sf(j, i) = K(gl2, gl2);
		}
	}
	//K_ss
	for (unsigned int i = 0; i < D_s.size(); i++) {
		for (unsigned int j = 0; j < D_s.size(); j++) {
			int gl1 = d[i];
			int gl2 = d[j];
			K_ss(i, j) = K(gl1, gl2);
		}
	}
	MatrixXd F_ext = F_f - K_fs * D_s;
	D_f = K_ff.inverse() * F_ext;
	F_s = K_sf * D_f + K_ss * D_s;
}
