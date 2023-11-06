import torch


def power_iteration(A, q0, niter=36, rtol=1e-05, atol=1e-08):
    """Compute the operator norm of the linear operator A."""

    with torch.no_grad():
        for k in range(niter):
            zk = A(q0)
            zk_norm = torch.sqrt(torch.vdot(zk.flatten(), zk.flatten()))
            q0 = zk / zk_norm

        Aqk = A(q0)
        op_norm = torch.sqrt(torch.vdot(Aqk.flatten(), Aqk.flatten()))

    return op_norm
