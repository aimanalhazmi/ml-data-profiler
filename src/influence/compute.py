from src.influence import LogisticInfluence
from src.model import train


def compute_influence(model, X, y):
    influencer = LogisticInfluence(model, X, y)
    pass
