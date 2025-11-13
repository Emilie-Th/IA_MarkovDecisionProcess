#Résolution du GridWorld 4×4 via Value Iteration
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

Policy=np.zeros(NState,dtype=int)

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
    

if __name__ == "__main__":
    
    #Exécute l'algorithme pour les 3 valeurs de Gamma
    for g in range(3):
    
        #V0 est initialisé à 0 car on ne connait rien de l'environnement
        V0=np.zeros(NState) 
        VOld=V0.copy()
        while True :
            
            VNew=VOld.copy()
            for s in range (NState):
                
                #pour les états terminaux on n’applique pas la mise à jour de Bellman
                if IsTerminal(s):
                    continue
                #on applique la mise à jour de Bellman
                else :
                    av=[]
                    for a in range (NAction):
                        av.append(R+Gamma[g]*VOld[NextState(s,a)])
                    VNew[s]=np.max(av)
                    Policy[s]=av.index(np.max(av)) 
            #vérification du critère d’arrêt
            if (np.max(np.abs(VNew-VOld))<=Theta):
                break
            VOld=VNew.copy()
            
        #Affichage
        print("----------------- Gamma = ",Gamma[g],"-----------------")
        np.set_printoptions(precision=2, floatmode='fixed') #permet d’afficher les décimales
        #remplacement des indices d'actions par des symboles pour un affichage plus visuel (indexation vectorisée proposée par NumPy)
        arrows = np.array(["↑", "→", "↓", "←"])
        grid = arrows[Policy.astype(int)]
        #masque pour afficher les états terminaux avec ‘T’
        mask = np.zeros(N*N, dtype=bool)
        mask[[0,15]]=True
        grid = np.where(mask, "T", grid)     #np.where(condition, valeur_si_true, valeur_si_false)
        print("Value Function V(s) : \n")
        print(VNew.reshape(N,N),"\n")
        print("Optimal Policy : \n")     
        print(grid.reshape(N, N))
        print("\n","\n")
        
        
        
                
            
    