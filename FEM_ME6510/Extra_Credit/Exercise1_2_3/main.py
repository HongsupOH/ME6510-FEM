import numpy as np
from shape_grad import shape_gradient
from gauss_integration import gauss
from Jacobi import Jacobian
from volume import volume

if __name__=="__main__":
    print("Exercise 2")
    node_info = np.array([[[0,0],[2,0],[2,2],[0,2]],
                          [[0,0],[1,0],[1,1],[0,1]],
                          [[0,0],[4,0],[4,4],[0,4]],
                          [[0,0],[2,0],[2,1],[0,1]],
                          [[0,0],[1,0],[1,2],[0,2]],
                          [[0,0],[1,0],[2,2],[0,1]]])

    gausses = np.array([1,2])
    for gauss_order in gausses:
        print("Gauss order is: {}".format(gauss_order))
        gp,gw = gauss(gauss_order)
        for node_coors in node_info:
            print('Node coordinate is:')
            print(node_coors)
            for i in range(len(gp)):
                for j in range(len(gp)):
                    xi = np.array([gp[i],gp[j]])
                    J,J_inv,detJ = Jacobian(node_coors,xi)
                    print('At xi is {},'.format(xi))
                    print("[J]:")
                    print(J)
                    print("|J|:")
                    print(detJ)
            print("")
            
    print("")
    print("Exercise 3")
    h = 1
    node_info2 = np.array([[[0,0],[1,0],[1,1],[0,1]],
                           [[0,0],[1,0],[0.25,0.25],[0,1]],
                           [[0,0],[1,0],[-0.25,-0.25],[0,1]]])
    
    gausses = np.array([1,2,3])
    for gauss_order in gausses:
        print("Gauss order is: {}".format(gauss_order))
        for node_coors in node_info2:
            print("Node coordinate:")
            print(node_coors)
            v = volume(node_coors,h,gauss_order)
            print('ANS: ',v)
    
    
