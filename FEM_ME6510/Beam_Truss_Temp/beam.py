import numpy as np

def K_e(E,I,L):
    K_ele = (E*I/(L**3))*np.array([[12,6*L,-12,6*L],
                                   [6*L,4*(L**2),-6*L,2*(L**2)],
                                   [-12,-6*L,12,-6*L],
                                   [6*L,2*(L**2),-6*L,4*(L**2)]])
    
    return K_ele

def GlobK(num_ele,num_node,Es,Is,Ls):
    dofs = 2*num_node
    K = np.zeros((dofs,dofs))
    glbInd = [0,1,2,3]
    for ele in range(num_ele):
        E = Es[ele]
        I = Is[ele]
        L = Ls[ele]
        K_ele = K_e(E,I,L)
        print(K_ele)
        print("")
        for loc1,glb1 in enumerate(glbInd):
            for loc2,glb2 in enumerate(glbInd):
                K[glb1,glb2]+= K_ele[loc1,loc2]
        
        cnt = 0
        while cnt<2:
            glbInd.pop(0)
            cnt+=1
        
        cnt2 = 0
        newInd = glbInd[-1]+1
        while cnt2<2:
            glbInd.append(newInd)
            newInd+=1
            cnt2+=1

    return K

def solve(f,s,f_f,d_s,K):
    f_ext = f_f - np.dot(K[np.ix_(f,s)],d_s)
    d_f = np.linalg.solve(K[np.ix_(f,f)],f_ext)
    f_s = np.dot(K[np.ix_(s,f)],d_f) + np.dot(K[np.ix_(s,s)],d_s)
    return d_f,f_s

def main(Es,Is,Ls,num_ele,num_node,f,s,f_f,d_s):
    print("Glob K:")
    K = GlobK(num_ele,num_node,Es,Is,Ls)
    print(K)
    print("")
    print("Answer for unknow factors:")
    d_f,f_s = solve(f,s,f_f,d_s,K)
    print("Unknown deformation:")
    print("d_{f}: ",d_f)
    print("Unknown force:")
    print("f_{s}: ",f_s)
    
if __name__=="__main__":
    #Define variables
    Es = [30*10**6,10*10**6]
    Is = [400,200]
    Ls = [60,60]
    num_ele = 2
    num_node = 3
    f = np.array([2,3,4,5])
    s = np.array([0,1])
    f_f = np.array([-5000,0,-5000,0])
    d_s = np.array([0,0])
    #Call main function
    main(Es,Is,Ls,num_ele,num_node,f,s,f_f,d_s)
    print("Test")
    Es = [30*10**6,30*10**6]
    Is = [62.5,87.5]
    Ls = [25,25]
    num_ele = 2
    num_node = 3
    f = np.array([0,1,2,3])
    s = np.array([4,5])
    f_f = np.array([0,1000,0,0])
    d_s = np.array([0,0])
    #Call main function
    main(Es,Is,Ls,num_ele,num_node,f,s,f_f,d_s)
    print("Test")
    
    
