class InfluenceFunctionBase:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn

    def _get_inv_hessian(self):
        raise NotImplementedError()

    def _grad_loss(self, x, y):
        raise NotImplementedError()

    def get_influence(self, x_test, y_test):
        raise NotImplementedError()
