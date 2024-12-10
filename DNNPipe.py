"""
@author: Woobean Seo
@affiliation: Real-Time Operating System Laboratory, Seoul National University
@contact: wbseo@redwood.snu.ac.kr
@date: 2024-12-10
"""
from itertools import combinations
import math
import time
import numpy as np

INF = math.inf
def DNNPipe(N, C, M, U, E, R, conditions=['lb', 'ub']): 
    T_ideal = sum(E)/sum(C)
    T_ub = T_ideal + max(E)/min(C)
    D = [i for i in range(1, N+1)]
    D_sub_total = sum([list(map(list, combinations(D, i))) for i in range(N + 1)], []) 
    iteration_cnt = 0
    P, MST, ST_, R_ = {}, {}, {}, {}
    for i in range(0, U+1):
        for S in D_sub_total:
            P[(i, tuple(S))] = []  
            MST[(i, tuple(S))] = 0 if i == 0 else INF 
        if i == 0: continue
        for n in D:
          for j in range(i, U+1):
              ST_[(i, j, n)] = sum(E[i-1:j])/C[n-1]
              R_[(i, j)] = sum(R[i-1:j])
    P_opt = []
    T_max_opt = INF
    for S in D_sub_total:
        for u in set(D) - set(S):
            new_S = tuple(sorted(list(S)+[u]))
            for i in range(len(S)+1, U+1):
                for j in range(i, U+1): 
                    T_s = ST_[(i, j, u)] 
                    if MST[(i-1, tuple(S))] == INF or R_[(i, j)]>M[u-1] : continue
                    if 'ub' in conditions and T_s > T_ub: break
                    if 'lb' in conditions and j != U:
                        if (T_s + E[j]/C[u-1]) < T_ideal: continue
                    T_max = max(MST[(i-1, tuple(S))], T_s)
                    if T_max <= MST[(j, new_S)]:
                        MST[(j, new_S)]  = T_max
                        P[(j, new_S)] = P[i-1, tuple(S)] + [(i, j, u)] 
                        iteration_cnt += 1
                        if j == U and T_max<=T_max_opt:
                            T_max_opt = T_max
                            P_opt = P[(j, new_S)]
    return P_opt