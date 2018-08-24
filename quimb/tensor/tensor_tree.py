import itertools
import math

import numpy as np

from .tensor_core import rand_uuid, Tensor, TensorNetwork
from .tensor_1d import TensorNetwork1D, TensorNetwork1DVector


class TreeTN(TensorNetwork1DVector, TensorNetwork1D, TensorNetwork):
    r"""Construct a Tree Tensor Network (TTN).

               __________O__________
              /          |          \
           __O__       __O__       __O__
          /  |  \     /  |  \     /  |  \
         O   O   O   O   O   O   O   O   O
        /|\ /|\ /|\ /|\ /|\ /|\ /|\ /|\ /|\

    """

    _EXTRA_PROPS = ("_site_ind_id", "_site_tag_id", "cyclic")

    def __init__(self, n, arrays=None, base=2, dangle=False,
                 site_ind_id="k{}", site_tag_id="I{}", **tn_opts):

        # short-circuit for copying MERA
        if isinstance(n, TreeTN):
            super().__init__(n)
            for ep in TreeTN._EXTRA_PROPS:
                setattr(self, ep, getattr(n, ep))
            return

        self._site_ind_id = site_ind_id
        self._site_tag_id = site_tag_id
        self.cyclic = True

        nlayers = round(math.log(n, base))

        if isinstance(arrays, np.ndarray):
            arrays = (arrays,)

        arrays = itertools.cycle(arrays)

        def gen_ttn_tensors():
            u_ind_id = site_ind_id

            for i in range(nlayers):

                # index id connecting to layer below
                l_ind_id = u_ind_id
                # index id connecting to layer above
                u_ind_id = rand_uuid() + "_{}"

                # number of tensor sites in this layer
                eff_n = n // base ** i

                for j in range(0, eff_n, base):

                    eff_sites = [((j + b) % eff_n) for b in range(base)]
                    lix = tuple(map(l_ind_id.format, eff_sites))
                    uix = (u_ind_id.format(j // base),)
                    inds = lix + uix

                    tags = {"_LAYER{}".format(i)}
                    if i == 0:
                        for site in eff_sites:
                            tags.add(site_tag_id.format(site))

                    yield Tensor(next(arrays), inds, tags=tags)

        super().__init__(
            gen_ttn_tensors(), check_collisions=False, structure=site_tag_id
        )

        # tag the TTN with the 'causal-cone' of each site
        for i in range(nlayers):
            for j in range(n):
                # get tensors in layer above
                for t in self.select_neighbors(j):
                    if f"_LAYER{i + 1}" in t.tags:
                        t.add_tag(f"I{j}")

    @classmethod
    def rand(cls, n, base=2, invar=False, **ttn_opts):
        pass
