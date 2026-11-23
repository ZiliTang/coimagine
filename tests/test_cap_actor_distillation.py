import torch
from torch.nn.functional import cross_entropy
from experiments.planar_tasks.authoring_v2.cap_actor_distillation import set_loss


def test_single_target_matches_masked_cross_entropy():
    logits = torch.tensor([[1.5, 8., 0.], [4., 2., 3.]])
    valid = torch.tensor([[True, False, True], [True, True, True]])
    target = torch.tensor([[False, False, True], [True, False, False]])
    expected = cross_entropy(logits.masked_fill(~valid, -torch.inf), torch.tensor([2, 0]), reduction='none')
    torch.testing.assert_close(set_loss(logits, valid, target), expected)


def test_set_loss_rewards_probability_mass_and_ignores_padding():
    logits = torch.log(torch.tensor([[2., 3., 100., 5.]])).requires_grad_()
    valid = torch.tensor([[True, True, False, True]])
    winners = torch.tensor([[True, True, False, False]])
    loss = set_loss(logits, valid, winners)
    torch.testing.assert_close(loss, -torch.tensor([.5]).log())
    loss.sum().backward()
    assert logits.grad[0,2] == 0 and (logits.grad[0,:2]<0).all() and logits.grad[0,3]>0
    torch.testing.assert_close(set_loss(logits, valid, valid), torch.zeros(1))
