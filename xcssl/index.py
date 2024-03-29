import itertools

from .rasterizer import make_rasterizer


class RandomIndex:
    def __init__(self, encoding, rasterizer_kwargs, phenotypes):
        self._encoding = encoding

        self._rasterizer = make_rasterizer(self._encoding.obs_space,
                                           rasterizer_kwargs)

        self._phenotype_count_map = self._init_phenotype_count_map(phenotypes)

        (self._phenotype_grid_cells_map,
         self._grid_cell_phenotypes_map) = self._gen_maps(
             self._encoding, self._rasterizer,
             phenotype_set=self._phenotype_count_map.keys())

    @classmethod
    def from_scratch(cls, encoding, rasterizer_kwargs):
        # no phenotypes passed to init
        return cls(encoding, rasterizer_kwargs, phenotypes=[])

    @classmethod
    def from_existing(cls, encoding, rasterizer_kwargs, phenotypes):
        # given phenotypes passed to init
        return cls(encoding, rasterizer_kwargs, phenotypes)

    def _init_phenotype_count_map(self, phenotypes):
        phenotype_count_map = {}

        for phenotype in phenotypes:
            try:
                phenotype_count_map[phenotype] += 1
            except KeyError:
                phenotype_count_map[phenotype] = 1

        return phenotype_count_map

    def _gen_maps(self, encoding, rasterizer, phenotype_set):
        # each 'grid cell' is a string of the form "(e_1, e_2, ..., e_k)"
        # where each element e_i specifies the bin idx for dimension i
        grid_cell_phenotypes_map = {}
        k = rasterizer.num_grid_dims
        b = rasterizer.num_bins_per_grid_dim

        # init the grid cell phenotypes map with empty sets for all b**k
        # combos of bins
        for grid_cell_bin_combo_tup in itertools.product(
                *itertools.repeat(tuple(range(0, b)), k)):
            grid_cell_phenotypes_map[str(grid_cell_bin_combo_tup)] = set()

        phenotype_grid_cells_map = {}
        for phenotype in phenotype_set:

            aabb = phenotype.monkey_patch_and_return_aabb(encoding)

            grid_cells_covered_iter = rasterizer.rasterize_aabb(aabb)
            grid_cells_covered = []

            for grid_cell_bin_combo_tup in grid_cells_covered_iter:

                grid_cell = str(grid_cell_bin_combo_tup)

                grid_cells_covered.append(grid_cell)
                (grid_cell_phenotypes_map[grid_cell]).add(phenotype)

            phenotype_grid_cells_map[phenotype] = grid_cells_covered

        return (phenotype_grid_cells_map, grid_cell_phenotypes_map)

    def gen_sparse_phenotype_matching_map(self, obs):
        sparse_phenotype_matching_map = {}

        grid_cell = str(self._rasterizer.rasterize_obs(obs))

        for phenotype in self._grid_cell_phenotypes_map[grid_cell]:

            sparse_phenotype_matching_map[
                phenotype] = self._rasterizer.match_idxd_aabb(phenotype.aabb, obs)

        return sparse_phenotype_matching_map

    def try_add_phenotype(self, phenotype):
        try:
            self._phenotype_count_map[phenotype] += 1
        except KeyError:
            self._phenotype_count_map[phenotype] = 1
            self._add_phenotype(phenotype)

    def _add_phenotype(self, addee):
        aabb = addee.monkey_patch_and_return_aabb(self._encoding)

        grid_cells_covered_iter = self._rasterizer.rasterize_aabb(aabb)
        grid_cells_covered = []

        for grid_cell_bin_combo_tup in grid_cells_covered_iter:

            grid_cell = str(grid_cell_bin_combo_tup)

            grid_cells_covered.append(grid_cell)
            (self._grid_cell_phenotypes_map[grid_cell]).add(addee)

        self._phenotype_grid_cells_map[addee] = grid_cells_covered

    def try_remove_phenotype(self, phenotype):
        count = self._phenotype_count_map[phenotype]
        count -= 1

        if count == 0:
            del self._phenotype_count_map[phenotype]
            self._remove_phenotype(phenotype)
        else:
            self._phenotype_count_map[phenotype] = count

    def _remove_phenotype(self, removee):
        grid_cells_covered = self._phenotype_grid_cells_map[removee]
        for grid_cell in grid_cells_covered:
            (self._grid_cell_phenotypes_map[grid_cell]).remove(removee)

        del self._phenotype_grid_cells_map[removee]
