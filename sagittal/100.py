N_list=list()

for i in range(10) :
    N = int(input())
    if N%42 not in N_list:
        N_list.append(N%42)
        
print(len(N_list))