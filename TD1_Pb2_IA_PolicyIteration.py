#Résolution du GridWorld 4×4 via Policy Iteration
import numpy as np

N = 4 #taille de la grille
NState = 16  #nombre d’états possibles
NAction = 4 #nombre d’actions possibles
R = -1 #récompense
Theta = 1e-4 #critère d’arrêt
Gamma = np.array([1,0.9,0.8]) #facteur d'actualisation


# Actions possibles
Up = 0 
Right = 1
Down = 2
Left = 3

def IsTerminal (n):
    return (n==0 or n==15)
        
def NextState (s,a):
    Row = s//N
    Column = s%N
    if a == 0 and Row > 0:
        Row -=1
    elif a == 1 and Column < N-1:
        Column +=1
    elif  a == 2 and Row < N-1:
        Row +=1
    elif a == 3 and Column > 0:
            Column -=1
    return Row*N+Column

def PolicyEvaluation(nState,gamma,theta,policy,v):
    VOld=v.copy()
    while True:
        VNew=VOld.copy()
        for s in range (nState):
            
            #pour les états finaux on n’applique pas la mise à jour de Bellman
            if IsTerminal(s):
                continue
            else :
                APolicy=policy[s]
                VNew[s]=R+gamma*VOld[NextState(s, APolicy)]
        #vérification du critère d’arrêt
        if (np.max(np.abs(VNew-VOld))<=Theta):
            return VNew
        VOld=VNew.copy()
    
def PolicyImprovement(nState,policy,v,gamma):
    
    for s in range (nState):
        
        #pour les états terminaux on n’applique pas la mise à jour de Bellman
        if IsTerminal(s):
            continue
        else :
            OldAction=policy[s]
            BestAction=OldAction
            BestValue=-1e9
            for a in range (NAction):
                Value=R+gamma*v[NextState(s, a)]
                if (Value > BestValue):
                    BestValue=Value
                    BestAction=a
            policy[s]=BestAction
            if(BestAction!=OldAction):
                return policy, False
    return policy, True
        

if __name__ == "__main__":

    #Exécute l’algorithme pour les 3 valeurs de Gamma
    for g in range(3):
    
        #Initialisation de la policy 
        #/!\ pour Gamma = 1 une policy contenant des boucles (self-loops) peut ne pas converger
        #-> ici choix d'une policy sans boucle mais plusieurs autres initialisations auraient fonctionné
        Policy=np.array([1,1,1,2,0,3,2,2,2,3,2,2,1,1,1,1])
        V0=np.zeros(NState) 
            
        while True :
            V=PolicyEvaluation(NState,Gamma[g],Theta,Policy,V0)
            Policy , PolicyStable = PolicyImprovement(NState,Policy,V,Gamma[g])
            if PolicyStable==True:
                break
                
        #Affichage
        print("----------------- Gamma = ",Gamma[g],"-----------------")
        np.set_printoptions(precision=2, floatmode='fixed') #permet d'afficher les décimales
        #remplacement des indices d'actions par des symboles pour un affichage plus visuel (indexation vectorisée proposée par NumPy)
        arrows = np.array(["↑", "→", "↓", "←"])
        grid = arrows[Policy.astype(int)]
        #masque pour afficher les états terminaux avec ‘T’
        mask = np.zeros(N*N, dtype=bool)
        mask[[0,15]]=True
        grid = np.where(mask, "T", grid)     #np.where(condition, valeur_si_true, valeur_si_false)
        print("Value Function V(s) : \n")
        print(V.reshape(N,N),"\n")
        print("Optimal Policy : \n")     
        #print(Policy.reshape(N,N),"\n")
        print(grid.reshape(N, N))
        print("\n","\n")