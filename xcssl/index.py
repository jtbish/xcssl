import itertools
import bisect

from .rasterizer import make_rasterizer


class RandomIndex:
    def __init__(self, encoding, rasterizer_kwargs, int_id_clfr_map):
        self._encoding = encoding

        self._rasterizer = make_rasterizer(self._encoding.obs_space,
                                           rasterizer_kwargs)

        (self._int_id_grid_cells_map,
         self._grid_cell_sorted_int_ids_map) = self._gen_maps(
             self._encoding, self._rasterizer, int_id_clfr_map)

    @classmethod
    def from_scratch(cls, encoding, rasterizer_kwargs):
        # empty map passed to init
        return cls(encoding, rasterizer_kwargs, int_id_clfr_map={})

    @classmethod
    def from_existing(cls, encoding, rasterizer_kwargs, int_id_clfr_map):
        # given map passed to init
        return cls(encoding, rasterizer_kwargs, int_id_clfr_map)

    def _gen_maps(self, encoding, rasterizer, int_id_clfr_map):
        # each 'grid cell' is a string of the form "(e_1, e_2, ..., e_k)"
        # where each element e_i specifies the bin idx for dimension i
        grid_cell_sorted_int_ids_map = {}
        k = rasterizer.num_grid_dims
        b = rasterizer.num_bins_per_grid_dim

        # init the grid cell int ids map with empty lists for all b**k
        # combos of bins
        for grid_cell_bin_combo_tup in itertools.product(
                *itertools.repeat(tuple(range(0, b)), k)):
            grid_cell_sorted_int_ids_map[str(grid_cell_bin_combo_tup)] = []

        int_id_grid_cells_map = {}
        for (int_id, clfr) in int_id_clfr_map.items():

            phenotype = clfr.condition.phenotype
            aabb = phenotype.monkey_patch_and_return_aabb(encoding)

            grid_cells_covered_iter = rasterizer.rasterize_aabb(aabb)
            grid_cells_covered = []

            for grid_cell_bin_combo_tup in grid_cells_covered_iter:

                grid_cell = str(grid_cell_bin_combo_tup)

                grid_cells_covered.append(grid_cell)
                # insert int_id into the grid cell int ids map, 
                # keeping sorted order
                bisect.insort(grid_cell_sorted_int_ids_map[grid_cell], int_id)

            int_id_grid_cells_map[int_id] = grid_cells_covered

        return (int_id_grid_cells_map, grid_cell_sorted_int_ids_map)

    def add(self, int_id, phenotype):
        aabb = phenotype.monkey_patch_and_return_aabb(self._encoding)

        grid_cells_covered_iter = self._rasterizer.rasterize_aabb(aabb)
        grid_cells_covered = []

        for grid_cell_bin_combo_tup in grid_cells_covered_iter:

            grid_cell = str(grid_cell_bin_combo_tup)

            grid_cells_covered.append(grid_cell)
            # insert int_id into the grid cell int ids map, 
            # keeping sorted order
            bisect.insort(self._grid_cell_sorted_int_ids_map[grid_cell], int_id)

        self._int_id_grid_cells_map[int_id] = grid_cells_covered

    def remove(self, int_id):
        grid_cells_covered = self._int_id_grid_cells_map[int_id]
        for grid_cell in grid_cells_covered:
            (self._grid_cell_sorted_int_ids_map[grid_cell]).remove(int_id)

        del self._int_id_grid_cells_map[int_id]

    def gen_match_set(self, int_id_clfr_map, obs):
        match_set = []

        grid_cell = str(self._rasterizer.rasterize_obs(obs))
        # since below for loop is in sorted int id order, will yield same match
        # set order as VanillaPopuation, and thus the rest of the stochastic
        # process will follow same RNG trajectory
        for int_id in self._grid_cell_sorted_int_ids_map[grid_cell]:

            clfr = int_id_clfr_map[int_id]
            if self._rasterizer.match_idxd_aabb(
                    aabb=clfr.condition.phenotype.aabb, obs=obs):
                match_set.add(clfr)

        return match_set
