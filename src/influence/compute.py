from src.influence import LogisticInfluence


def compute_influence(model, X, y):
    influencer = LogisticInfluence(model, X, y)
    pass