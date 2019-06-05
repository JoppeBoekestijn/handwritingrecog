#Made by: s295454
#NLP, week 2 exercise 6

import sys
import numpy as np

def main(argv):
    s1 = sys.argv[1]
    s2 = sys.argv[2]
    min_distance(s1,s2)

def min_distance(s1, s2):
    m = len(s1)
    n = len(s2)
    d = np.zeros((m+1,n+1))

    for i in range(1,m):
        d[i,0] = i
    for j in range(1,n):
        d[0,j] = j

    for j in range(1,n+1):
      for i in range(1,m+1):
          if(s1[i-1] == s2[j-1]):
            substitutionCost = 0
          else:
            substitutionCost = 2
          d[i, j] = min(d[i-1, j] + 1, #deletion
                             d[i, j-1] + 1,  #insertion
                             d[i-1, j-1] + substitutionCost)  #substitution
    print(d[m,n])

if __name__ == "__main__" :
	main(sys.argv)
