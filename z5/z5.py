#created by Desert_Fox_Fenek & Ciemny99

import numba
import numpy
import time
from numba import jit

def print_alns_only(ALIGNMENTS):
    print('Wszystkie przyrównani: ' + str(len(ALIGNMENTS)))
    print('Wynik punktacji: '+str(ALIGNMENTS[0][3][0][1])+'\n')
    
    for elem in ALIGNMENTS:
        print(elem[0]+'\n'+elem[1]+'\n')
    return

#@jit(parallel=True, nogil=True)
def find_each_patch(c_i,c_j,path = ''):
    global ALN_PATHWAYS
    i = c_i
    j = c_j

    if i == 0 and j == 0:
        ALN_PATHWAYS.append(path)
        return 2

    dir_t = len(MATRIX[i][j][1])

    while dir_t <= 1:
        n_dir = MATRIX[i][j][1][0] if (i != 0 and j != 0) else (1 if i == 0 else (3 if j==0 else 0))
        path = path + str(n_dir)

        if n_dir == 1:
            j = j - 1
        elif n_dir == 2:
            j = j - 1
            i = i - 1
        elif n_dir == 3:
            i = i - 1
            dir_t = len(MATRIX[i][j][1])

        if i == 0 and j == 0:
            ALN_PATHWAYS.append(path)
            return 3

    if dir_t > 1:
        for dir_c in range (dir_t):
            n_dir = MATRIX[i][j][1][dir_c] if (i != 0 and j != 0) else (1 if i == 0 else (3 if j==0 else 0))
            tmp_path = path + str(n_dir)

            if n_dir == 1:
                n_i = i
                n_j=j-1
            elif n_dir == 2:
                n_i=i-1
                n_j=j-1
            elif n_dir == 3:
                n_i=i-1
                n_j = j
            find_each_patch(n_i,n_j,tmp_path)
    return len(ALN_PATHWAYS)

#@jit(parallel=True, nogil=True)
def point(SEQUENCE_1,SEQUENCE_2,GAP_CHARACTER):
    global ALN_PATHWAYS
    MATRIX_ROW_N = len(SEQUENCE_1) + 1
    MATRIX_COLUMN_N = len(SEQUENCE_2) + 1

    MATCH_SCORE = 6
    MISMATCH_SCORE = 3
    GAP_SCORE = -6

    MATRIX = [[[[None] for i in range(2)] for i in range(MATRIX_COLUMN_N)] for i in range(MATRIX_ROW_N)]
    for i in range(MATRIX_ROW_N):
        MATRIX[i][0] = [GAP_SCORE*i,[]]
    for j in range(MATRIX_COLUMN_N):
        MATRIX[0][j] = [GAP_SCORE*j,[]]
    for i in range(1,MATRIX_ROW_N):
        for j in range(1,MATRIX_COLUMN_N):
            score = MATCH_SCORE if (SEQUENCE_1[i-1] == SEQUENCE_2[j-1]) else MISMATCH_SCORE
            h_val = MATRIX[i][j-1][0] + GAP_SCORE
            d_val = MATRIX[i-1][j-1][0] + score
            v_val = MATRIX[i-1][j][0] + GAP_SCORE
            o_val = [h_val, d_val, v_val]
            MATRIX[i][j] = [max(o_val), [i+1 for i,v in enumerate(o_val) if v==max(o_val)]]
    OVERALL_SCORE = MATRIX[i][j][0]
    #print(MATRIX)
    return OVERALL_SCORE, i,j , MATRIX

def check_seq(SEQ_1,SEQ_2):
    alph = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'J', 'Z', 'X']
    p = 1
    while (True):
        for i in range (len(SEQ_1)):
            if SEQ_1[i] not in alph:
                p = 0
                break

        if p == 0: break

        for i in range (len(SEQ_2)):
            if SEQ_2[i] not in alph:
                p = 0
                break
        break
    if p == 0:
        print("Błędna sekwencja")
        input()

def save(ALIGNMENTS):
    data = open("Porównanie genów.txt", "w")
    i = 0
    data.write('Wynik punktacji: '+str(ALIGNMENTS[0][3][0][1])+'\n')
    for elem in ALIGNMENTS:
        i += 1
        data.write(str(i)+' para\n'+elem[0]+'\n'+elem[1]+'\n\n')
    data.close()

SEQUENCE_1 = 'MENEREKQVYLAKLSEQTERYDEMVEAMKKVAQLDVELTVEERNLVSVGYKNVIGARRASWRILSSIEQKEESKGNDENVKRLKNYRKRVEDELAKVCNDILSVIDKHLIPSSNAVESTVFFYKMKGDYYRYLAEFSSGAERKEAADQSLEAYKAAVAAAENGLAPTHPVRLGLALNFSVFYYEILNSPESACQLAKQAFDDAIAELDSLNEESYKDSTLIMQLLRDNLTLWTSDLNEEGDERTKGADEPQDEV'
SEQUENCE_2 = 'MENERAKQVYLAKLNEQAERYDEMVEAMKKVAALDVELTIEERNLLSVGYKNVIGARRASWRILSSIEQKEESKGNEQNAKRIKDYRTKVEEELSKICYDILAVIDKHLVPFATSGESTVFYYKMKGDYFRYLAEFKSGADREEAADLSLKAYEAATSSASTELSTTHPIRLGLALNFSVFYYEILNSPERACHLAKRAFDEAIAELDSLNEDSYKDSTLIMQLLRDNLTLWTSDLEEGGK'

GAP_CHARACTER = '-'
ALN_PATHWAYS = []
ALIGNMENTS = []

check_seq(SEQUENCE_1,SEQUENCE_2)

score, i, j, MATRIX = point(SEQUENCE_1,SEQUENCE_2,GAP_CHARACTER)
l_i = i
l_j = j

start = time.clock()
tot_aln = find_each_patch(i,j)
stop = time.clock()

aln_count = 0

for elem in ALN_PATHWAYS:
    i = l_i-1
    j = l_j-1
    side_aln = ''
    top_aln = ''
    step = 0
    aln_info = []
    for n_dir_c in range(len(elem)):
        n_dir = elem[n_dir_c]
        score = MATRIX[i+1][j+1][0]
        step = step + 1
        aln_info.append([step,score,n_dir])
        if n_dir == '2':
            side_aln = side_aln + SEQUENCE_1[i]
            top_aln = top_aln + SEQUENCE_2[j]
            i=i-1
            j=j-1
        elif n_dir == '1':
            side_aln = side_aln + GAP_CHARACTER
            top_aln = top_aln + SEQUENCE_2[j]
            j=j-1
        elif n_dir == '3':
            side_aln = side_aln + SEQUENCE_1[i]
            top_aln = top_aln + GAP_CHARACTER
            i=i-1
    aln_count = aln_count + 1
    ALIGNMENTS.append([top_aln[::-1],side_aln[::-1],elem,aln_info,aln_count])
    s = ([top_aln[::-1],side_aln[::-1],elem,aln_info,aln_count])

save(ALIGNMENTS)

print("Czas obliczeń programu: {0:02f}s".format(stop-start))
print_alns_only(ALIGNMENTS)
