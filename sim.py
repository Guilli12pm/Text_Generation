import torch

def block_diag(m, device=None):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n, device=device).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))

# A roughly drop-in replacement to torchquantum.QuantumDevice
# for free-fermion circuits using the covariance matrix representation.
class FFQuantumDevice:
    def __init__(self, qubits: int, batch_size: int, device: torch.device = None):
        # Create the covariance matrix for the all-zero state:
        m = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], device=device)
        m = torch.block_diag(*[m]*qubits)
        self.cov = m[None, :, :].repeat(batch_size, 1, 1)

        self.qubits = qubits
        self.batch_size = batch_size

    # This method performs exp(h)@cov@exp(-h) where h is all-zeros except h[i,j] = a and h[j,i] = -a
    def conj_givens(self, i: int, j: int, theta: torch.Tensor):
        # In this case, H = exp(h) is the identity matrix, except that
        # H[i,i] = H[j,j] = cos(a), H[i,j] = sin(a), H[j,i] = -sin(a)
        c = torch.cos(theta)
        s = torch.sin(theta)
        
        # Multiply self.cov by H. Since this is the identity
        # matrix except for two rows, we can just update those rows directly
        # (the extra None dimension is to handle the batching correctly)
        a = (c[:, None] - 1) * self.cov[:, i, :] + s[:, None] * self.cov[:, j, :]
        b = -s[:, None] * self.cov[:, i, :] + (c[:, None] - 1) * self.cov[:, j, :]
        self.cov = self.cov + a[:, None, :] * (torch.arange(self.cov.shape[1], device=self.cov.device) == i)[None, :, None]
        self.cov = self.cov + b[:, None, :] * (torch.arange(self.cov.shape[1], device=self.cov.device) == j)[None, :, None]

        # Multiply self.cov by exp(-h) = H^T. This is the same except columns instead of rows.
        a = (c[:, None] - 1) * self.cov[:, :, i] + s[:, None] * self.cov[:, :, j]
        b = -s[:, None] * self.cov[:, :, i] + (c[:, None] - 1) * self.cov[:, :, j]
        self.cov = self.cov + a[:, :, None] * (torch.arange(self.cov.shape[2], device=self.cov.device) == i)[None, None, :]
        self.cov = self.cov + b[:, :, None] * (torch.arange(self.cov.shape[2], device=self.cov.device) == j)[None, None, :]

    # Apply an RZ gate on qubit q with angle theta
    # (theta is a vector of angles over the batch dimension)
    def rz(self, q: int, theta: torch.Tensor):
        # Z rotation is just a conjugation within one 'block', so i = 2q, j = 2q+1 
        self.conj_givens(2*q, 2*q+1, theta)

    # Apply an RXX gate on qubits q and q+1 with angle theta
    # (theta is a vector of angles over the batch dimension)
    def rxx(self, q: int, theta: torch.Tensor):
        # XX rotation is a conjugation between 'block's so i=2q+1, j=2(q+1)
        self.conj_givens(2*q+1, 2*(q+1), theta)

    def rz_layer(self, theta: torch.Tensor):
        c = torch.cos(theta)
        s = torch.sin(theta)
        ms = torch.stack((
            torch.stack((c, s), dim=2),
            torch.stack((-s, c), dim=2)
        ), dim=2)
        exph = block_diag(ms, device=theta.device)
        self.cov = exph @ self.cov @ exph.mT

    def rxx_layer(self, theta: torch.Tensor):
        theta2 = torch.nn.functional.pad(theta, (1, 1), mode='constant')
        c = torch.cos(theta2)
        s = torch.sin(theta2)
        ms = torch.stack((
            torch.stack((c, s), dim=2),
            torch.stack((-s, c), dim=2)
        ), dim=2)
        exph = block_diag(ms, device=theta.device)[:, 1:-1, 1:-1]
        self.cov = exph @ self.cov @ exph.mT

    # Get the Z expectation value of all qubits. Returns a matrix of batch x qubits.
    def z_exp_all(self) -> torch.Tensor:
        return torch.diagonal(self.cov, offset=1, dim1=1, dim2=2)[:, ::2]

# Now for some tests:
if __name__ == "__main__":
    import torchquantum
    import random
    import tqdm

    # Test that torch can autodiff:
    p = torch.nn.Parameter(torch.rand((1, 10), requires_grad=True), requires_grad=True)
    m = FFQuantumDevice(10, 1)
    m.rxx_layer(torch.rand((1, 9)))
    m.rz_layer(p)
    m.rxx_layer(torch.rand((1, 9)))
    v = m.z_exp_all()[0, 0]
    v.backward()
    assert p.grad is not None

    # Test 1000 random circuits
    print("Testing 1000 random circuits...")
    for _ in tqdm.trange(1000):
        q_statevector = torchquantum.QuantumDevice(10, bsz=32)
        q_freefermion = FFQuantumDevice(10, 32)
        # Apply 100 random gates:
        for _ in range(10):
            theta = torch.rand((q_freefermion.batch_size, 10))
            if random.random() < 0.5:
                for q in range(10):
                    q_statevector.rz(q, theta[:, q])
                q_freefermion.rz_layer(theta)
            else:
                for q in range(9):
                    q_statevector.rxx([q,q+1], theta[:, q])
                q_freefermion.rxx_layer(theta[:, :-1])
        expval_sv = torchquantum.expval_joint_analytical(q_statevector, 'ZIIIIIIIII')
        expval_ff = q_freefermion.z_exp_all()[:,0]
        assert torch.allclose(expval_sv, expval_ff, rtol=1.0, atol=1e-6)
    print("Tests passed successfully")
