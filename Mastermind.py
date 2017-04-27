import math
import random
from copy import deepcopy

def split(colors, combos, depth):
    if depth == 1:
        return combos
    else:
        colors_L = deepcopy(colors)
        combos_L = deepcopy(combos)
        for combo_i in range(len(combos_L)): #For every combo
            thisElement_s = combos_L[combo_i]
            temp_L = [thisElement_s]*len(colors_L) #Split into 3 duplicates
            for temp_i in range(len(temp_L)):
                temp_L[temp_i] += colors_L[temp_i]
            combos_L[combo_i] = temp_L
        result_L = []
        for sublist_L in combos_L:
            for element_s in sublist_L:
                result_L += [element_s]
        return split(colors, result_L, depth-1)

def generate_combinations(colors_L, slots_i):
    return split(colors_L, colors_L, slots_i)

def generate_code(colors, slots):
    return ''.join([random.choice(colors) for i in range(slots)])
    #return [random.choice(colors) for i in range(slots)]

def score(guess, answer):    
    if len(guess) != len(answer):
        print('DIM MISMATCH')
        return
    guess_L = deepcopy(guess)
    answer_L = deepcopy(answer)
    blackPegs_i = 0
    for i in range(len(guess_L)):
        if guess_L[i] == answer_L[i]:
            blackPegs_i += 1
            guess_L[i] = 0
            answer_L[i] = 0
    guess_L = [a for a in guess_L if a != 0]
    answer_L = [a for a in answer_L if a != 0]
    whitePegs_i = 0
    for s in guess_L:
        if s in answer_L:
            whitePegs_i += 1
            answer_L[answer_L.index(s)] = 0
    return [blackPegs_i, whitePegs_i]

def scores_list(combos, guess):
    scoreList_L = []
    for combo_s in combos:
        scoreList_L += [[combo_s, score(list(combo_s), list(guess))]]
    return scoreList_L

def all_scores(combos, guess):
    scoreList_L = []
    for combo_s in combos:
        scoreList_L += [score(list(combo_s), list(guess))]
    return scoreList_L

def naive_solve(combos_L, answer_L, score_L=[]):
    guesses_i = 0
    for guess_L in combos_L:
        guesses_i += 1
        if guess_L == answer_L:
            return [guess_L, guesses_i]

def removed(scores, hint):
    removed = []
    for item in scores:
        if item[1] == hint:
            removed += [item[0]]
    return removed

def display(removed, answer):
    print(str(answer), len(removed))
    print('-'*len(answer))
    for item in removed:
        print(str(item))

def solve(combos_L, answer_s, guessesTaken_i=1):
    combos_L = deepcopy(combos_L)
    if answer_s not in combos_L:
        return "Error: Answer not in combos"
    guess_s = random.choice(combos_L)
    print("Guess #"+str(guessesTaken_i)+": "+guess_s)
    if len(combos_L) == 1:
        return "You Won!!! Answer is "+str(combos_L[0])
    hint_L = score(list(guess_s), list(answer_s))
    guessesTaken_i += 1
    scores_L = all_scores(combos_L, guess_s)
    removing_L = []
    for item_i in range(len(scores_L)):
        if scores_L[item_i] != hint_L:
            removing_L += [combos_L[item_i]]
            combos_L[item_i] = 0
    print("Removing as canidates..."+str(removing_L))
    while 0 in combos_L:
        combos_L.remove(0)
    return solve(combos_L, answer_s, guessesTaken_i)


slots = 4
colors = ['red','blue','green']
size = math.pow(len(colors), slots)
combos = generate_combinations(colors, slots)
answer = generate_code(colors, slots)

print(solve(combos, answer)) #Takes combos, solves, then checks against given answer
