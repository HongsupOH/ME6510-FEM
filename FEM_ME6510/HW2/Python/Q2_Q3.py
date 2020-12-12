import numpy as np

class Homework2(object):
    def truss_stiffness(self,theta,k):
        theta_r = theta*np.pi/180
        k_locd = np.array([[k,0,-k,0],
                           [0,0,0,0],
                           [-k,0,k,0],
                           [0,0,0,0]])
        C = np.cos(theta_r)
        S = np.sin(theta_r)
        Tr = np.array([[C,S,0,0],
                       [-S,C,0,0],
                       [0,0,C,S],
                       [0,0,-S,C]])
        self.Tr = Tr
        k_globd = np.dot(Tr.T,np.dot(k_locd,Tr))
        return k_globd
    
    def truss_force(self,theta,E,L,d1x,d1y,d2x,d2y):
        theta_r = theta*np.pi/180
        C = np.cos(theta_r)
        S = np.sin(theta_r)
        return (E/L)*np.dot(np.array([-C,-S,C,S]),np.array([d1x,d1y,d2x,d2y]))


if __name__=="__main__":
    #Question2
    print("Exercise2")
    q2 = Homework2()
    #part-(a)
    q2_a = q2.truss_stiffness(45,45e5)
    print("(a). Global direction K: ")
    print(q2_a)
    #part-(b)
    q2_b = q2.truss_stiffness(120,1e6)
    print("(b). Global direction K: ")
    print(q2_b)
    #part-(c)
    q2_c = q2.truss_stiffness(-30,28)
    print("(c). Global direction K: ")
    print(q2_c)
    #part-(d)
    q2_d = q2.truss_stiffness(-20,14)
    print("(d). Global direction K: ")
    print(q2_d)
    print("")
    #Question3
    print("Exercise3")
    theta_a = 45
    E_a = 30e6
    A_a = 2
    L_a = 60
    d1x_a,d1y_a,d2x_a,d2y_a = 0,0,0.01,0.02
    stress_a = q2.truss_force(theta_a,E_a,L_a,d1x_a,d1y_a,d2x_a,d2y_a)
    print("(a). Stress is: ",stress_a)
    theta_b = 30
    E_b = 210
    A_b = 3e2
    L_b = 3e3
    d1x_b,d1y_b,d2x_b,d2y_b = 0.25,0,1,0
    stress_b = q2.truss_force(theta_b,E_b,L_b,d1x_b,d1y_b,d2x_b,d2y_b)
    print("(b). Stress is: ",stress_b)
                            
