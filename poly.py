import itertools

def polynomial_terms(variables, max_degree):
    terms = []
    for degree in range(max_degree + 1):
        term_combinations = itertools.combinations_with_replacement(variables, degree)
        terms.extend(['*'.join(term) for term in term_combinations])
    return terms

variables = ['x', 'y']
max_degree = 4

polynomial_terms = polynomial_terms(variables, max_degree)
print(polynomial_terms)
