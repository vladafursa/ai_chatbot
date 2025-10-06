import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

#variables' definition
size = ctrl.Antecedent(np.arange(0, 11, 1), 'size')
socialBehaviour = ctrl.Antecedent(np.arange(0, 11, 1), 'socialBehaviour')
fit = ctrl.Antecedent(np.arange(0, 11, 1), 'fit')
fur = ctrl.Antecedent(np.arange(0, 11, 1), 'fur')
domestication = ctrl.Consequent(np.arange(0, 101, 1), 'domestication')

#membership function for size (triangle)
size['small'] = fuzz.trimf(size.universe, [0, 0, 5])
size['medium'] = fuzz.trimf(size.universe, [0, 5, 10])
size['large'] = fuzz.trimf(size.universe, [5, 10, 10])

#membership function for sozial behaviour (triangle)
socialBehaviour['aggressive'] = fuzz.trimf(socialBehaviour.universe, [0, 0, 5])
socialBehaviour['neutral'] = fuzz.trimf(socialBehaviour.universe, [0, 5, 10])
socialBehaviour['friendly'] = fuzz.trimf(socialBehaviour.universe, [5, 10, 10])

#membership function for fit (triangle)
fit['thin'] = fuzz.trimf(fit.universe, [0, 0, 5])
fit['average'] = fuzz.trimf(fit.universe, [0, 5, 10])
fit['fat'] = fuzz.trimf(fit.universe, [5, 10, 10])

#membership function for fur (triangles)
fur['coarse'] = fuzz.trimf(fur.universe, [0, 0, 5])
fur['soft'] = fuzz.trimf(fur.universe, [5, 10, 10])

#membership function for target wildness (triangle)
domestication["wild"] = fuzz.trimf(domestication.universe, [0, 0, 50])
domestication["domestic"] = fuzz.trimf(domestication.universe, [50, 100, 100])

#fuzzy rules definitions
rule1 = ctrl.Rule(socialBehaviour['friendly'] | fur['soft'], domestication["domestic"])
rule2 = ctrl.Rule(size['small'] & fit['fat'], domestication["domestic"])
rule3 = ctrl.Rule(socialBehaviour['aggressive'] & size['large'], domestication["wild"])
rule4 = ctrl.Rule(fur['coarse'] & (fit['thin'] | fit['average']), domestication["wild"])
rule5 = ctrl.Rule(socialBehaviour['neutral'] & fur['soft'], domestication["domestic"])

domesticationCtrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
domesticationRes = ctrl.ControlSystemSimulation(domesticationCtrl)


def getDomestication(size, socialBehaviour, fit, fur):
    domesticationRes.input["size"] = size
    domesticationRes.input["socialBehaviour"] = socialBehaviour
    domesticationRes.input["fit"] = fit
    domesticationRes.input["fur"] = fur
    domesticationRes.compute()
    return domesticationRes.output["domestication"]

#for input checks
def getInteger(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value<0 or value > 10:
                print("Input should be integer between 0 and 10")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter an integer.")