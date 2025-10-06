from nltk.sem import Expression
from nltk.inference import ResolutionProver
read_expr = Expression.fromstring
import pandas

LOGICAL_KB_FILENAME = "logical-kb.csv"
#  Initialise Knowledgebase.
'''reading knowledge base from a file and appending array with logical expression converted from each line
(string converted into Expression object)'''
def getKB():
    kb = []
    try:
        data = pandas.read_csv(LOGICAL_KB_FILENAME, header=None)
        [kb.append(read_expr(row)) for row in data[0]]
    except FileNotFoundError:
        print("The file doesn't exist")
    except Exception as e:
        print(f"An exception occured: {e}")
    return kb


def checkForContradictions(kb):
    for expression in kb:
        negatedExpression = read_expr(f"-{str(expression)}")
        # check if negation is provable
        if ResolutionProver().prove(negatedExpression, kb, verbose = False):
            print("Initial KB had contradiction")
            exit(1)

def appendKB(object, subject, kb):
    assumptionString = subject + ' (' + object + ')'  # creating string to parse into logical expression
    assumptionExpression = read_expr(assumptionString)
    negatedAssumption = read_expr(f'-({assumptionString})')  # making it negative to proof by contradiction
    if ResolutionProver().prove(negatedAssumption, kb, verbose=False):
        print("Contradiction!")
    else:
        print("OK, I will remember that " + object + " is " + subject)
        kb.append(assumptionExpression)

def checkFact(object, subject, kb):
    expr = read_expr(subject + '(' + object + ')') # creating string to parse into logical expression
    answer = ResolutionProver().prove(expr, kb, verbose=False)
    if answer:
        print('Correct.')
    else:
        negatedExpr = read_expr('-' + subject + '(' + object + ')')
        if ResolutionProver().prove(negatedExpr, kb, verbose=False):  # if negation can be proven then it is incorrect
            print("Incorrect.")
        else:  # uncertain (can be true or is not know, but whas proven that it is not true earlier
            print("Sorry, I don't know.")





