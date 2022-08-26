import torch

class Sampler(object):
    def __init__(self, model, dataset):
        model.eval()
        self.set = set
        self.dataset = dataset
        self.final_activations = []
        self.penultimate_activations = []
        self.set_activations_from_model(model)

    def set_activations_from_model(self, model):
        model.eval()
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=100, shuffle=False, sampler=None,
                                             batch_sampler=None, num_workers=10)
        s = 0
        for img in loader:
            s += 100
            final_acts, penultimate_acts = model(img[0].cuda())
            self.final_activations.extend(final_acts.detach().cpu().numpy())
            self.penultimate_activations.extend(penultimate_acts.detach().cpu().numpy())


    def get_subset(self):
        raise NotImplementedError
