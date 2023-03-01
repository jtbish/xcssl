import numpy as np

from .rasterizer import make_rasterizer


class CoverageMap:
    def __init__(self, encoding, rasterizer_kwargs, phenotypes):
        self._encoding = encoding

        self._rasterizer = make_rasterizer(self._encoding.obs_space,
                                           rasterizer_kwargs)

        self._phenotype_count_map = self._init_phenotype_count_map(phenotypes)

        self._phenotype_aabb_map = self._init_phenotype_aabb_map(
            self._encoding, phenotype_set=self._phenotype_count_map.keys())

        (self._phenotype_grid_cells_map,
         self._grid_cell_phenotypes_map) = self._gen_maps(
             self._encoding, self._rasterizer, self._phenotype_aabb_map)

    @classmethod
    def from_scratch(cls, encoding, rasterizer_kwargs):
        # no phenotypes passed to init
        return cls(encoding, rasterizer_kwargs, phenotypes=[])

    @classmethod
    def from_phenotypes(cls, encoding, rasterizer_kwargs, phenotypes):
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

    def _init_phenotype_aabb_map(self, encoding, phenotype_set):
        return {
            phenotype: encoding.make_phenotype_aabb(phenotype)
            for phenotype in phenotype_set
        }

    def _gen_maps(self, encoding, rasterizer, phenotype_aabb_map):

        phenotype_grid_cells_map = {}
        grid_cell_phenotypes_map = np.empty(shape=rasterizer.num_grid_cells,
                                            dtype="object")
        for grid_cell in range(0, rasterizer.num_grid_cells):
            grid_cell_phenotypes_map[grid_cell] = set()

        for (phenotype, aabb) in phenotype_aabb_map.items():

            grid_cells_covered_iter = rasterizer.rasterize_aabb(aabb)
            grid_cells_covered = []

            for grid_cell_bin_combo in grid_cells_covered_iter:

                grid_cell = rasterizer.convert_grid_cell_bin_combo_to_dec(
                    grid_cell_bin_combo)

                grid_cells_covered.append(grid_cell)

                (grid_cell_phenotypes_map[grid_cell]).add(phenotype)

            phenotype_grid_cells_map[phenotype] = tuple(grid_cells_covered)

        return (phenotype_grid_cells_map, grid_cell_phenotypes_map)

    def gen_sparse_phenotype_matching_map(self, obs):
        sparse_phenotype_matching_map = {}

        grid_cell = self._rasterizer.rasterize_obs(obs)

        for phenotype in self._grid_cell_phenotypes_map[grid_cell]:

            aabb = self._phenotype_aabb_map[phenotype]

            sparse_phenotype_matching_map[
                phenotype] = self._rasterizer.match_idxd_aabb(aabb, obs)

        return sparse_phenotype_matching_map

    def try_add_phenotype(self, phenotype):
        try:
            self._phenotype_count_map[phenotype] += 1
        except KeyError:
            self._phenotype_count_map[phenotype] = 1
            do_add = True
        else:
            do_add = False

        if do_add:
            self._add_phenotype(phenotype)

    def _add_phenotype(self, addee):

        aabb = self._encoding.make_phenotype_aabb(addee)
        self._phenotype_aabb_map[addee] = aabb

        grid_cells_covered_iter = self._rasterizer.rasterize_aabb(aabb)
        grid_cells_covered = []

        for grid_cell_bin_combo in grid_cells_covered_iter:

            grid_cell = self._rasterizer.convert_grid_cell_bin_combo_to_dec(
                grid_cell_bin_combo)

            grid_cells_covered.append(grid_cell)

            (self._grid_cell_phenotypes_map[grid_cell]).add(addee)

        self._phenotype_grid_cells_map[addee] = tuple(grid_cells_covered)

    def try_remove_phenotype(self, phenotype):
        count = self._phenotype_count_map[phenotype]
        count -= 1

        do_remove = (count == 0)

        if do_remove:
            del self._phenotype_count_map[phenotype]
            self._remove_phenotype(phenotype)

        else:
            self._phenotype_count_map[phenotype] = count

    def _remove_phenotype(self, removee):
        del self._phenotype_aabb_map[removee]

        grid_cells_covered = self._phenotype_grid_cells_map[removee]
        del self._phenotype_grid_cells_map[removee]

        for grid_cell in grid_cells_covered:
            (self._grid_cell_phenotypes_map[grid_cell]).remove(removee)
