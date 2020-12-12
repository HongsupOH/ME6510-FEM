import numpy as np

def K_e(A,L,h,k):
    K_ele = ((A*k)/L)*np.array([[1,-1],[-1,1]]) + A*h*np.array([[0,0],[0,1]])
    return K_ele

def GlobK(num_ele,num_node,As,hs,Ls,ks,nodes):
    dofs = num_node
    K = np.zeros((dofs,dofs))
    for ele in range(num_ele):
        A = As[ele]
        h = hs[ele]
        L = Ls[ele]
        k = ks[ele]
        glbInd = nodes[ele]
        K_ele = K_e(A,L,h,k)
        for loc1,glb1 in enumerate(glbInd):
            for loc2,glb2 in enumerate(glbInd):
                K[glb1,glb2]+= K_ele[loc1,loc2]
        
    return K

def solve(f,s,f_f,d_s,K):
    f_ext = f_f - np.dot(K[np.ix_(f,s)],d_s)
    d_f = np.linalg.solve(K[np.ix_(f,f)],f_ext)
    f_s = np.dot(K[np.ix_(s,f)],d_f) + np.dot(K[np.ix_(s,s)],d_s)
    return d_f,f_s

def main(As,hs,Ls,ks,num_ele,num_node,f,s,f_f,d_s,nodes):
    Temp = np.array([0,0,0,0])
    heat = np.array([0,0,0,0])
    print("Glob K:")
    K = GlobK(num_ele,num_node,As,hs,Ls,ks,nodes)
    print(K)
    print("")
    print("Answer for unknow factors:")
    print(np.linalg.inv(K).dot(np.array([14,0,0,3466.68])))
    d_f,f_s = solve(f,s,f_f,d_s,K)
    print("Unknown temperature:")
    print("d_{f}: ",d_f)
    print("Unknown heat flux:")
    print("f_{s}: ",f_s)
    print("")
    for ind,sind in enumerate(s):
        Temp[sind] = d_s[ind]
        heat[sind] = f_s[ind]
    for ind,find in enumerate(f):
        Temp[find] = d_f[ind]
        heat[find] = f_f[ind]
    print("Temperature vector:")
    print("d: ",Temp)
    print("Heatflux vector:")
    print("f: ",heat)

if __name__=="__main__":
    print("problem2")
    #Define variables
    As = [0.5*0.5, 0.5*0.5, 0.5*0.5]
    Ls = [0.1,0.2,0.12]
    hs = [0,0,45]
    ks = [0.12,0.071,0.72]
    num_ele = 3
    num_node = 4
    f = np.array([1,2,3])
    s = np.array([0])
    f_f = np.array([0,0,3466.68])
    d_s = np.array([523.15])
    nodes = np.array([[0,1],[1,2],[2,3]])
    #Call main function
    main(As,hs,Ls,ks,num_ele,num_node,f,s,f_f,d_s,nodes)
        
